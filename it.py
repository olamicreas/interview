#!/usr/bin/env python3
"""
app.py — Interview Assistant (patched)

Changes in this patched version:
- Remote-only login (AUTH_SERVER_URL must be set).
- Uses langdetect to filter transcripts to English only (fast).
- Avoids sending very short/noisy captures (VAD + min-length checks).
- Recorder can select a loopback/system-audio device by substring via AUDIO_DEVICE env var.
- GUI status won't be overwritten while recording (Recording label persists until release).
- Tuned default trimming minimum to reduce spurious short uploads.

Note: install new dependency:
    pip install langdetect==1.0.9
"""

import os
import io
import queue
import threading
import json
import time
import argparse
import sys
import wave
import requests
import sounddevice as sd
import webrtcvad
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import uuid
import datetime
import getpass

from rich.console import Console
from rich.markdown import Markdown
from openai import OpenAI
from pynput import keyboard  # optional global keyboard listener

# PySide6 GUI
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit,
    QDialog, QFormLayout, QLineEdit, QMessageBox
)
from PySide6.QtCore import QTimer, Qt

# language detection
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0  # deterministic

console = Console()

# ------------------ Configuration (speed-focused) ------------------
# audio
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
BYTES_PER_SAMPLE = 2
FRAME_DURATION_MS = int(os.environ.get("FRAME_DURATION_MS", "10"))  # must be 10/20/30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * BYTES_PER_SAMPLE

# latency tuning
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "3"))
PRE_ROLL_MS = float(os.environ.get("PRE_ROLL_MS", "80"))
PRE_ROLL_FRAMES = max(1, int(PRE_ROLL_MS / FRAME_DURATION_MS))

# increased min keep to avoid tiny noises being accepted
TRIM_PRE_ROLL_MS = int(os.environ.get("TRIM_PRE_ROLL_MS", "60"))
TRIM_PRE_ROLL_FRAMES = max(1, int(TRIM_PRE_ROLL_MS / FRAME_DURATION_MS))
TRIM_MIN_KEEP_MS = int(os.environ.get("TRIM_MIN_KEEP_MS", "160"))  # increased from 80

# SECURITY: use environment variables for API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-nano")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-1")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are my live interview assistant. For every interviewer question, produce a spoken-style answer "
    "as the candidate in first person. Reply in 2–3 short bullet points using the STAR method "
    "(Situation/Task, Action, Result). Be concise, conversational, and suitable to say aloud in an interview."
)

# Whisper / retry tuning
WHISPER_MAX_RETRIES = int(os.environ.get("WHISPER_MAX_RETRIES", "1"))
WHISPER_BACKOFF_BASE = float(os.environ.get("WHISPER_BACKOFF_BASE", "0.12"))
ALLOW_WHISPER_FALLBACK = os.environ.get("ALLOW_WHISPER_FALLBACK", "false").lower() in ("1", "true", "yes")

# LLM speed/verbosity
STREAM_MAX_OUTPUT_TOKENS = int(os.environ.get("STREAM_MAX_OUTPUT_TOKENS", "64"))
FALLBACK_MAX_OUTPUT_TOKENS = int(os.environ.get("FALLBACK_MAX_OUTPUT_TOKENS", "96"))
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "minimal")
VERBOSITY = os.environ.get("OPENAI_VERBOSITY")
HISTORY_MAX_ITEMS = int(os.environ.get("HISTORY_MAX_ITEMS", "4"))

if FRAME_DURATION_MS not in (10, 20, 30):
    console.print("[red]FRAME_DURATION_MS must be 10, 20, or 30 (webrtcvad requirement).[/red]")
    sys.exit(1)

if not OPENAI_API_KEY:
    console.print("[bold red]Error:[/bold red] OPENAI_API_KEY env var not set. Export your key and retry.")
    sys.exit(1)

# Session/auth configuration (remote-only)
AUTH_SERVER_URL = os.environ.get("AUTH_SERVER_URL")  # must be set to remote auth endpoint
if not AUTH_SERVER_URL:
    console.print("[bold red]Error:[/bold red] AUTH_SERVER_URL must be set for remote-only login.")
    sys.exit(1)

# allow specifying a loopback audio device substring (e.g., "Stereo Mix", "BlackHole", "VB-Audio")
AUDIO_DEVICE_NAME = os.environ.get("AUDIO_DEVICE", "")

# English detection / filter thresholds
MIN_TRANSCRIPT_CHARS = int(os.environ.get("MIN_TRANSCRIPT_CHARS", "4"))
MIN_EN_CONF = float(os.environ.get("MIN_EN_CONF", "0.80"))

SESSION_DURATION_MIN = int(os.environ.get("SESSION_DURATION_MIN", "120"))  # default session lifetime

# ---------------------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY)

# persistent HTTP session for Whisper uploads
http_session = requests.Session()
http_session.headers.update({"Authorization": f"Bearer {OPENAI_API_KEY}"})

# ffmpeg / encoding config
FFMPEG_BIN = shutil.which("ffmpeg")
console.print(f"[dim]ffmpeg {'found' if FFMPEG_BIN else 'not found'} on PATH — "
              f"{'using Opus encoding' if FFMPEG_BIN else 'falling back to WAV uploads'}[/dim]")

# executor for encoding/upload with modest concurrency
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("ENCODE_WORKERS", "3")))
DEFAULT_OPUS_BITRATE_K = int(os.environ.get("OPUS_BITRATE_K", "16"))  # smaller upload = faster
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT_SEC", "6"))

# Queues and locks
_question_q = queue.Queue()     # transcribed questions for LLM
_ui_q = queue.Queue()           # (question, normalized_answer) -> GUI consumer
_history_lock = threading.Lock()
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

PUSH_TO_TALK = False   # global listener if requested
push_recording = False
push_buffer = bytearray()
push_lock = threading.Lock()

# ------------------ Session Manager (remote-only) ------------------
class SessionManager:
    """
    Session manager (remote-only):
    - Uses AUTH_SERVER_URL for authentication. No local or GitHub fallback.
    """
    def __init__(self):
        self.token: Optional[str] = None
        self.username: Optional[str] = None
        self.expires_at: Optional[datetime.datetime] = None
        self.lock = threading.Lock()

    def _now(self):
        return datetime.datetime.utcnow()

    def is_active(self) -> bool:
        with self.lock:
            if not self.token or not self.expires_at:
                return False
            return self._now() < self.expires_at

    def remaining_seconds(self) -> int:
        with self.lock:
            if not self.expires_at:
                return 0
            delta = self.expires_at - self._now()
            return max(0, int(delta.total_seconds()))

    def clear(self):
        with self.lock:
            self.token = None
            self.username = None
            self.expires_at = None

    def login_remote(self, username: str, password: str) -> (bool, str):
        try:
            resp = requests.post(AUTH_SERVER_URL, json={"username": username, "password": password}, timeout=8)
            # Expecting 200 with JSON {"token": "...", "expires_in": 3600}
            if resp.status_code == 200:
                j = resp.json()
                token = j.get("token")
                expires_in = j.get("expires_in")
                if not token:
                    return False, "Auth response missing token."
                if isinstance(expires_in, (int, float)) and expires_in > 0:
                    expires_at = self._now() + datetime.timedelta(seconds=int(expires_in))
                else:
                    expires_at = self._now() + datetime.timedelta(minutes=SESSION_DURATION_MIN)
                with self.lock:
                    self.username = username
                    self.token = token
                    self.expires_at = expires_at
                return True, "Logged in (remote)."
            elif resp.status_code in (401, 403):
                try:
                    msg = resp.json().get("error") or resp.json().get("message") or f"Auth failed: {resp.status_code}"
                except Exception:
                    msg = f"Auth failed: {resp.status_code}"
                return False, msg
            else:
                return False, f"Auth failed: HTTP {resp.status_code}"
        except requests.RequestException as e:
            return False, f"Auth request failed: {e}"
        except Exception as e:
            return False, f"Auth error: {e}"

    def login(self, username: str, password: str) -> (bool, str):
        # Remote-only login
        ok, msg = self.login_remote(username, password)
        return ok, msg

session_mgr = SessionManager()

# ------------------ Utilities ------------------
def trim_history():
    with _history_lock:
        if len(conversation_history) > HISTORY_MAX_ITEMS:
            conversation_history[:] = [conversation_history[0]] + conversation_history[-(HISTORY_MAX_ITEMS-1):]

def enforce_star_bullets(text: str, max_bullets: int = 3) -> str:
    if not text or not text.strip():
        return "[No answer returned by model]"
    text = re.sub(r'\s+', ' ', text.strip())
    lines = [ln.strip() for ln in re.split(r'[\r\n]+', text) if ln.strip()]
    bullet_like = [ln for ln in lines if re.match(r'^[\-\u2022\*]\s+', ln) or ln.lower().startswith("situation") or ln.lower().startswith("action") or ln.lower().startswith("result")]
    if bullet_like:
        out = []
        for ln in bullet_like[:max_bullets]:
            ln = re.sub(r'^[\-\u2022\*]\s*', '', ln)
            out.append(f"• {ln}")
        if len(out) == 1:
            out.append("• Action: Briefly describe what you did.")
        return "\n".join(out)
    sents = re.split(r'(?<=[.?!])\s+', text)
    sents = [s.strip().rstrip('.!?') for s in sents if s.strip()]
    labels = ["Situation/Task", "Action", "Result"]
    bullets = []
    for i in range(3):
        if i < len(sents):
            content = sents[i]
        else:
            content = ""
        content = re.sub(r'\b(I am|I\'m|I’m) an AI assistant\b', 'I', content, flags=re.IGNORECASE)
        if len(content) > 220:
            content = content[:217].rsplit(' ', 1)[0] + "..."
        if content:
            bullets.append(f"• {labels[i]}: {content}")
    if len(bullets) == 1:
        bullets.append("• Action: Briefly describe what you did.")
    return "\n".join(bullets[:max_bullets])

# ------------------ Recorder ------------------
class LiveRecorder:
    """
    RawInputStream opened only while recording to reduce overhead.
    Blocksize slightly reduced for lower latency.
    Accepts a device name substring to choose a specific input (e.g., loopback device).
    """
    def __init__(self, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * 0.03), dtype='int16', channels=1, device_name: str = None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype = dtype
        self.channels = channels
        self.stream = None
        self._running = False
        self.device_name = device_name

    def _callback(self, indata, frames, time_info, status):
        if status:
            console.print(f"[yellow]Audio status: {status}[/yellow]")
        data = bytes(indata)
        with push_lock:
            push_buffer.extend(data)

    def _resolve_device(self):
        if not self.device_name:
            return None
        try:
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                if self.device_name.lower() in d['name'].lower():
                    return idx
        except Exception:
            pass
        return None

    def start(self):
        if self._running:
            return
        with push_lock:
            push_buffer.clear()
        device_idx = self._resolve_device()
        try:
            self.stream = sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype,
                channels=self.channels,
                callback=self._callback,
                device=device_idx  # None selects default
            )
            self.stream.start()
            self._running = True
        except Exception as e:
            console.print(f"[red]Failed to open audio input (device={self.device_name} idx={device_idx}): {e}[/red]")
            raise

    def stop(self):
        if not self._running:
            return b""
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self._running = False
        with push_lock:
            data = bytes(push_buffer)
        return data

recorder = LiveRecorder(device_name=AUDIO_DEVICE_NAME)

# ------------------ Silence trimming ------------------
def trim_silence_pcm_fast(raw_pcm: bytes, pre_roll_frames=TRIM_PRE_ROLL_FRAMES, min_keep_ms=TRIM_MIN_KEEP_MS) -> bytes:
    if not raw_pcm or len(raw_pcm) < FRAME_BYTES * 2:
        return raw_pcm
    frames = [raw_pcm[i:i+FRAME_BYTES] for i in range(0, len(raw_pcm), FRAME_BYTES) if len(raw_pcm[i:i+FRAME_BYTES]) == FRAME_BYTES]
    if not frames:
        return raw_pcm
    speech_flags = []
    for f in frames:
        try:
            speech_flags.append(vad.is_speech(f, SAMPLE_RATE))
        except Exception:
            speech_flags.append(True)
    first = None
    last = None
    for idx, v in enumerate(speech_flags):
        if v:
            first = idx
            break
    for idx in range(len(speech_flags)-1, -1, -1):
        if speech_flags[idx]:
            last = idx
            break
    if first is None or last is None:
        return raw_pcm
    start = max(0, first - pre_roll_frames)
    end = min(len(frames)-1, last + pre_roll_frames)
    trimmed_frames = frames[start:end+1]
    trimmed = b"".join(trimmed_frames)
    min_keep_bytes = int((SAMPLE_RATE * (min_keep_ms / 1000.0)) * BYTES_PER_SAMPLE)
    if len(trimmed) < min_keep_bytes:
        return raw_pcm
    return trimmed

# ------------------ Encoding ------------------
def encode_to_opus_webm_bytes(raw_pcm: bytes, bitrate_k: int = DEFAULT_OPUS_BITRATE_K, timeout: int = FFMPEG_TIMEOUT) -> Optional[io.BytesIO]:
    if not FFMPEG_BIN:
        return None
    cmd = [
        FFMPEG_BIN,
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-b:a", f"{bitrate_k}k",
        "-vbr", "off",
        "-frame_duration", "60",
        "-application", "audio",
        "-f", "webm",
        "pipe:1"
    ]
    try:
        proc = subprocess.run(cmd, input=raw_pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        if proc.returncode == 0 and proc.stdout:
            return io.BytesIO(proc.stdout)
        else:
            console.print(f"[yellow]ffmpeg encoding failed (rc={proc.returncode}). stderr: {proc.stderr.decode('utf8', errors='ignore')[:200]}[/yellow]")
            return None
    except subprocess.TimeoutExpired:
        console.print("[yellow]ffmpeg encoding timed out[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]ffmpeg encode exception: {e}[/red]")
        return None

# ------------------ Whisper uploads ------------------
def whisper_upload_bytesio_wav(raw_pcm_bytes: bytes, timeout=30):
    bio = io.BytesIO()
    try:
        with wave.open(bio, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw_pcm_bytes)
        bio.seek(0)
    except Exception as e:
        console.print(f"[red]Failed to build in-memory WAV: {e}[/red]")
        return None
    url = "https://api.openai.com/v1/audio/transcriptions"
    data = {"model": WHISPER_MODEL}
    for attempt in range(1, WHISPER_MAX_RETRIES + 1):
        try:
            files = {"file": ("audio.wav", bio, "audio/wav")}
            resp = http_session.post(url, data=data, files=files, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = WHISPER_BACKOFF_BASE * (2 ** (attempt - 1))
                console.print(f"[yellow]Whisper 429/quota. Retrying in {wait:.2f}s...[/yellow]")
                time.sleep(wait)
                continue
            else:
                console.print(f"[red]Whisper API error {resp.status_code}: {resp.text}[/red]")
                return None
        except requests.RequestException as e:
            console.print(f"[red]Network error to Whisper: {e}. Retrying...[/red]")
            time.sleep(WHISPER_BACKOFF_BASE * (2 ** (attempt - 1)))
    return None

def whisper_upload_fileobj_webm(fileobj: io.BytesIO, filename="audio.webm", content_type="audio/webm", timeout=30):
    fileobj.seek(0)
    url = "https://api.openai.com/v1/audio/transcriptions"
    data = {"model": WHISPER_MODEL}
    for attempt in range(1, WHISPER_MAX_RETRIES + 1):
        try:
            files = {"file": (filename, fileobj, content_type)}
            resp = http_session.post(url, data=data, files=files, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = WHISPER_BACKOFF_BASE * (2 ** (attempt - 1))
                console.print(f"[yellow]Whisper 429/quota. Retrying in {wait:.2f}s...[/yellow]")
                time.sleep(wait)
                continue
            else:
                console.print(f"[red]Whisper API error {resp.status_code}: {resp.text}[/red]")
                return None
        except requests.RequestException as e:
            console.print(f"[red]Network error to Whisper: {e}. Retrying...[/red]")
            time.sleep(WHISPER_BACKOFF_BASE * (2 ** (attempt - 1)))
    return None

# ------------------ Helpers: English detection and filters ------------------
def is_english_text(text: str) -> bool:
    """
    Return True if text is likely English with sufficient confidence.
    Conservative: short/incomplete text tends to be rejected.
    """
    if not text:
        return False
    text = text.strip()
    if len(text) < MIN_TRANSCRIPT_CHARS:
        return False
    # quick ASCII-letter check
    if not re.search(r'[A-Za-z]', text):
        return False
    try:
        langs = detect_langs(text)
        if not langs:
            return False
        top = langs[0]
        lang = getattr(top, 'lang', '')
        prob = float(getattr(top, 'prob', 0.0))
        return (lang == 'en') and (prob >= MIN_EN_CONF)
    except Exception:
        return False

# ------------------ Push-to-talk (non-blocking) ------------------
def process_push_buffer_and_queue(raw_bytes: bytes):
    if not raw_bytes:
        console.print("[yellow]Push-to-talk: no audio captured.[/yellow]")
        return

    # block uploads if session inactive
    if not session_mgr.is_active():
        console.print("[red]Session expired or not logged in — discarding capture. Please log in.[/red]")
        try:
            _ui_q.put_nowait(("__error__", "Session expired. Please log in."))
        except Exception:
            pass
        return

    trimmed_local = trim_silence_pcm_fast(raw_bytes)
    orig_len = len(raw_bytes)
    trimmed_len = len(trimmed_local)
    if trimmed_len < orig_len:
        console.print(f"[green]Trimmed audio: {orig_len} -> {trimmed_len} bytes ({(100*trimmed_len/orig_len):.0f}%) — encoding/uploading trimmed[/green]")
    else:
        console.print(f"[green]No trimming applied (uploading full capture: {orig_len} bytes)[/green]")

    # require a short minimum (avoid tiny noise)
    min_bytes = int((SAMPLE_RATE * 0.12) * BYTES_PER_SAMPLE)  # 120ms minimum
    if len(trimmed_local) < min_bytes:
        console.print("[yellow]Trimmed audio too short — likely noise. Discarding.[/yellow]")
        return

    def _job(trimmed_bytes: bytes):
        # try webm/opus first
        if FFMPEG_BIN:
            webm_bio = encode_to_opus_webm_bytes(trimmed_bytes, bitrate_k=DEFAULT_OPUS_BITRATE_K)
            if webm_bio:
                resp_json = whisper_upload_fileobj_webm(webm_bio, filename="audio.webm", content_type="audio/webm", timeout=FFMPEG_TIMEOUT+6)
                if resp_json and isinstance(resp_json, dict):
                    text = resp_json.get("text", "").strip()
                    if text:
                        # English filter
                        if not is_english_text(text):
                            console.print(f"[yellow]Discarding non-English or too-short transcript: '{text}'[/yellow]")
                        else:
                            console.print(f"\n[bold cyan]Transcribed question (push-to-talk):[/bold cyan] {text}")
                            if text.lower().strip() in ("exit", "quit", "bye"):
                                _question_q.put(None)
                                os._exit(0)
                            _question_q.put(text)
                            return
                    else:
                        console.print("[yellow]Whisper returned empty transcript for webm upload.[/yellow]")
                else:
                    console.print("[red]WebM upload/transcription failed — falling back to WAV upload.[/red]")

        # fallback WAV
        resp_json = whisper_upload_bytesio_wav(trimmed_bytes, timeout=20)
        if resp_json and isinstance(resp_json, dict):
            text = resp_json.get("text", "").strip()
            if text:
                if not is_english_text(text):
                    console.print(f"[yellow]Discarding non-English or too-short transcript: '{text}'[/yellow]")
                else:
                    console.print(f"\n[bold cyan]Transcribed question (push-to-talk WAV fallback):[/bold cyan] {text}")
                    if text.lower().strip() in ("exit", "quit", "bye"):
                        _question_q.put(None)
                        os._exit(0)
                    _question_q.put(text)
                    return
            else:
                console.print("[yellow]Whisper returned empty transcript for WAV upload.[/yellow]")

        if ALLOW_WHISPER_FALLBACK:
            # fallback prompt if you want something always
            _question_q.put("Can you describe a time you led a team?")

    EXECUTOR.submit(_job, trimmed_local)

# ------------------ LLM worker ------------------
def extract_text_from_response_object(resp_obj):
    try:
        t = getattr(resp_obj, "output_text", None)
        if t:
            return t
    except Exception:
        pass
    try:
        if isinstance(resp_obj, dict):
            if resp_obj.get("output_text"):
                return resp_obj.get("output_text")
            out = resp_obj.get("output")
            if isinstance(out, list) and out:
                for item in out:
                    if isinstance(item, dict):
                        content = item.get("content") or item.get("message") or item.get("output")
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    if block.get("type") == "output_text" and block.get("text"):
                                        return block.get("text")
                                    if block.get("text"):
                                        return block.get("text")
            choices = resp_obj.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                if isinstance(c0, dict) and c0.get("text"):
                    return c0.get("text")
        return str(resp_obj)
    except Exception:
        return str(resp_obj)

def llm_worker(dummy_mode: bool = False):
    while True:
        question = _question_q.get()
        if question is None:
            try:
                _ui_q.put_nowait((None, None))
            except Exception:
                pass
            break

        # block processing if session expired
        if not session_mgr.is_active():
            console.print("[red]Session expired while waiting to process question — skipping LLM call.[/red]")
            try:
                _ui_q.put_nowait((question, "[Session expired. Please log in.]"))
            except Exception:
                pass
            continue

        with _history_lock:
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": question})

        console.print("[magenta]Querying AI (final answer only)...[/magenta]")

        assistant_text = ""
        try:
            create_params = {
                "model": OPENAI_CHAT_MODEL,
                "input": messages,
                "max_output_tokens": FALLBACK_MAX_OUTPUT_TOKENS,
                "reasoning": {"effort": REASONING_EFFORT},
            }
            if VERBOSITY:
                create_params["verbosity"] = VERBOSITY
            resp = client.responses.create(**create_params)
            assistant_text = extract_text_from_response_object(resp)
        except Exception as e:
            console.print(f"[red]Responses.create() failed:[/red] {e}")
            assistant_text = f"[Error communicating with LLM: {e}]"

        normalized = enforce_star_bullets(assistant_text)

        with _history_lock:
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": normalized})
        trim_history()

        try:
            _ui_q.put_nowait((question, normalized))
        except Exception:
            pass

        console.print(Markdown(f"**AI answer (final):**\n\n{normalized}\n"))

# ------------------ Keyboard push-to-talk (global via pynput) ------------------
def on_press(key):
    global push_recording
    try:
        if PUSH_TO_TALK and (key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r):
            if not push_recording:
                with push_lock:
                    push_buffer.clear()
                try:
                    recorder.start()
                except Exception as e:
                    console.print(f"[red]Recorder start failed (keyboard): {e}[/red]")
                    return
                push_recording = True
                console.print("[green]Push-to-talk: recording... (hold Shift)[/green]")
    except Exception:
        pass

def on_release(key):
    global push_recording
    try:
        if PUSH_TO_TALK and (key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r):
            if push_recording:
                push_recording = False
                console.print("[green]Push-to-talk: released — sending audio...[/green]")
                raw = recorder.stop()
                t = threading.Thread(target=process_push_buffer_and_queue, args=(raw,), daemon=True)
                t.start()
    except Exception:
        pass

# ------------------ Login Dialog ------------------
class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.setFixedSize(360, 150)
        layout = QFormLayout(self)

        self.user_edit = QLineEdit()
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.Password)

        layout.addRow("Username:", self.user_edit)
        layout.addRow("Password:", self.pass_edit)

        self.status_lbl = QLabel("")
        layout.addRow(self.status_lbl)

        self.btn_login = QPushButton("Log in")
        self.btn_login.clicked.connect(self.attempt_login)
        layout.addRow(self.btn_login)

    def attempt_login(self):
        username = self.user_edit.text().strip()
        password = self.pass_edit.text().strip()
        if not username or not password:
            self.status_lbl.setText("Enter username and password.")
            return

        self.btn_login.setEnabled(False)
        ok, msg = session_mgr.login(username, password)
        self.btn_login.setEnabled(True)
        if ok:
            self.accept()
        else:
            self.status_lbl.setText(msg)

# ------------------ PySide6 GUI ------------------
class PushToTalkGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interview Assistant — Hold to Record (Fast)")
        self.setFixedSize(520, 380)

        # ensure the window receives keyboard events (Shift)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)

        self.status = QLabel("Not logged in")
        self.status.setStyleSheet("background-color: #333333; color: white; padding: 6px;")
        layout.addWidget(self.status)

        self.btn = QPushButton("Hold to Record")
        self.btn.setFixedHeight(70)
        self.btn.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; font-size: 16px;")
        layout.addWidget(self.btn)

        # prevent the button from grabbing keyboard focus (so window handles Shift)
        self.btn.setFocusPolicy(Qt.NoFocus)

        self.btn.pressed.connect(self.on_press)
        self.btn.released.connect(self.on_release)

        self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        self.answer_box.setPlaceholderText("Final AI answer will appear here (only final answer).")
        layout.addWidget(self.answer_box)

        # session-check timer (checks every 2s)
        self.session_timer = QTimer(self)
        self.session_timer.timeout.connect(self._session_tick)
        self.session_timer.start(2000)

        # more responsive polling interval for UI
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.poll_ui_queue)
        self.ui_timer.start(80)

        # internal guard to avoid multiple / overlapping login dialogs
        self._login_dialog_open = False

        # require login at start — scheduled slightly later so window finishes init
        QTimer.singleShot(50, self.ensure_logged_in)

    def ensure_logged_in(self):
        # Avoid opening multiple dialogs at once (race from timers)
        if session_mgr.is_active():
            self._update_status()
            return

        if self._login_dialog_open:
            return
        self._login_dialog_open = True
        try:
            dlg = LoginDialog(self)
            res = dlg.exec()
            if res != QDialog.Accepted or not session_mgr.is_active():
                QMessageBox.critical(self, "Login required", "Login is required to use the app.")
                QApplication.quit()
                return
            # success
            self._update_status()
        finally:
            self._login_dialog_open = False

    def _session_tick(self):
        # do not overwrite the status while actively recording
        global push_recording
        if push_recording:
            # keep the recording status shown until release
            return

        # previous behavior:
        if not session_mgr.is_active():
            self.set_status("Session expired — please log in", "#c0392b")
            if self._login_dialog_open:
                return
            self._login_dialog_open = True
            try:
                dlg = LoginDialog(self)
                res = dlg.exec()
                if res == QDialog.Accepted and session_mgr.is_active():
                    self._update_status()
                    self.btn.setEnabled(True)
                else:
                    self.btn.setEnabled(False)
                    self.set_status("Logged out. Please restart and log in.", "#c0392b")
            finally:
                self._login_dialog_open = False
        else:
            self._update_status()

    def _update_status(self):
        remaining = session_mgr.remaining_seconds()
        minutes = remaining // 60
        secs = remaining % 60
        self.set_status(f"Logged in as {session_mgr.username} — expires in {minutes}m{secs}s")

    def set_status(self, text, color=None):
        if color:
            self.status.setStyleSheet(f"background-color: {color}; color: white; padding: 6px;")
        else:
            self.status.setStyleSheet("background-color: #333333; color: white; padding: 6px;")
        self.status.setText(text)

    # GUI button handlers
    def on_press(self):
        if not session_mgr.is_active():
            self.set_status("Session expired — please log in", "#c0392b")
            return
        global push_recording
        with push_lock:
            push_buffer.clear()
        try:
            recorder.start()
        except Exception as e:
            self.set_status(f"Recorder error: {e}", "#c0392b")
            return
        push_recording = True
        self.set_status("Recording... release to send", "#b58900")
        self.answer_box.clear()

    def on_release(self):
        global push_recording
        if not session_mgr.is_active():
            self.set_status("Session expired — please log in", "#c0392b")
            return
        push_recording = False
        self.set_status("Encoding/uploading (fast)...", "#268bd2")
        raw = recorder.stop()
        threading.Thread(target=process_push_buffer_and_queue, args=(raw,), daemon=True).start()

    # keyboard handling while GUI window has focus: Shift hold -> record
    def keyPressEvent(self, event):
        try:
            if event.isAutoRepeat():
                return
        except Exception:
            pass
        if event.key() == Qt.Key_Shift:
            if not session_mgr.is_active():
                self.set_status("Session expired — please log in", "#c0392b")
                return
            global push_recording
            if not push_recording:
                with push_lock:
                    push_buffer.clear()
                try:
                    recorder.start()
                except Exception as e:
                    self.set_status(f"Recorder error: {e}", "#c0392b")
                    return
                push_recording = True
                self.set_status("Recording (Shift held)...", "#b58900")
                self.answer_box.clear()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        try:
            if event.isAutoRepeat():
                return
        except Exception:
            pass
        if event.key() == Qt.Key_Shift:
            global push_recording
            if push_recording:
                push_recording = False
                if not session_mgr.is_active():
                    self.set_status("Session expired — please log in", "#c0392b")
                    return
                self.set_status("Encoding/uploading (fast)...", "#268bd2")
                raw = recorder.stop()
                threading.Thread(target=process_push_buffer_and_queue, args=(raw,), daemon=True).start()
        super().keyReleaseEvent(event)

    def poll_ui_queue(self):
        try:
            while True:
                qitem = _ui_q.get_nowait()
                question, answer = qitem
                if question is None and answer is None:
                    return
                if question == "__error__":
                    self.set_status(answer, "#c0392b")
                    continue
                self.answer_box.setPlainText(answer)
                self.set_status("Ready")
        except queue.Empty:
            pass

# ------------------ Main & CLI login ------------------
def cli_login_prompt():
    print("Login required.")
    username = input("Username: ").strip()
    password = getpass.getpass("Password: ").strip()
    ok, msg = session_mgr.login(username, password)
    if not ok:
        print("Login failed:", msg)
        return False
    print("Login successful.")
    return True

def main():
    global PUSH_TO_TALK
    parser = argparse.ArgumentParser(description="Interview assistant (remote login + push-to-talk GUI)")
    parser.add_argument("--dummy", action="store_true", help="Enable dummy mode")
    parser.add_argument("--push-to-talk", action="store_true", help="Enable global hold-Shift push-to-talk (requires pynput)")
    parser.add_argument("--no-gui", action="store_true", help="Run without launching the PySide6 GUI (CLI only)")
    args = parser.parse_args()

    PUSH_TO_TALK = args.push_to_talk
    dummy_mode = args.dummy
    gui_enabled = not args.no_gui

    # start LLM worker thread
    llm_thread = threading.Thread(target=llm_worker, args=(dummy_mode,), daemon=True)
    llm_thread.start()

    # start global key listener if requested (pynput)
    if PUSH_TO_TALK:
        console.print("[cyan]Global push-to-talk enabled. Hold SHIFT to record (even when window unfocused) — if system allows.[/cyan]")
        try:
            key_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            key_listener.daemon = True
            key_listener.start()
        except Exception:
            console.print("[yellow]pynput global listener failed to start — GUI Shift will still work when window is focused.[/yellow]")

    if gui_enabled:
        app_qt = QApplication(sys.argv)
        window = PushToTalkGUI()

        # do NOT pre-show a login dialog here — the GUI will present it once via ensure_logged_in
        window.show()
        console.print("[cyan]GUI enabled. Hold the button or hold SHIFT (while window focused) to record questions; release to send.[/cyan]")

        try:
            app_qt.exec()
        except KeyboardInterrupt:
            pass
    else:
        # CLI mode: require login
        if not cli_login_prompt():
            console.print("[red]Login failed. Exiting.[/red]")
            return
        console.print("[green]Ready (CLI push-to-talk only). Ctrl+C to quit.[/green]")
        try:
            while True:
                if not session_mgr.is_active():
                    console.print("[red]Session expired. Please log in again.[/red]")
                    if not cli_login_prompt():
                        console.print("[red]Login failed. Exiting.[/red]")
                        break
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted by user. Exiting...[/bold yellow]")
            _question_q.put(None)
            _ui_q.put((None, None))
            time.sleep(0.2)

if __name__ == "__main__":
    main()

