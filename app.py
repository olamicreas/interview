from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from datetime import datetime, timedelta
import json
import os
import random
import string
from functools import wraps
import uuid # Imported for generating session tokens

app = Flask(__name__)

# --- CONFIGURATION ---
LICENSE_FILE = 'licenses.json'
ADMIN_PASSWORD = 'super-secure-admin-password' # !!! CHANGE THIS !!!
# --- END CONFIGURATION ---

# Initialize license file if it doesn't exist
if not os.path.exists(LICENSE_FILE):
    with open(LICENSE_FILE, 'w') as f:
        json.dump({"licenses": []}, f, indent=2)

def generate_password(length=10):
    """Generate a random alphanumeric password."""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def load_licenses():
    """Loads licenses from the JSON file."""
    try:
        with open(LICENSE_FILE, 'r') as f:
            data = json.load(f)
            return data.get('licenses', [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_licenses(licenses):
    """Writes licenses back to the JSON file."""
    with open(LICENSE_FILE, 'w') as f:
        json.dump({"licenses": licenses}, f, indent=2)

# Simple Admin Login Wrapper
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.cookies.get('admin_logged_in') != 'true':
            # For API routes, return a JSON error
            if request.path.startswith('/api/'):
                return jsonify({"error": "Admin login required"}), 401
            # For page routes, redirect to login (or serve the admin page with the modal)
            # Assuming client-side login is handled by the template
            return render_template('admin.html', admin_password=ADMIN_PASSWORD)
        return f(*args, **kwargs)
    return decorated_function

# --- FLASK ROUTES ---

@app.route('/')
def home():
    """Home route serving index.html."""
    return render_template('index.html')

@app.route('/download')
def download():
    """Download route serving download.html."""
    return render_template('download.html')

@app.route('/admin')
def admin():
    """Admin route serving admin.html with the login modal."""
    # The admin.html template will handle the client-side authentication
    return render_template('admin.html', admin_password=ADMIN_PASSWORD)

@app.route('/logout')
def logout():
    """Clears the login cookie and redirects to admin page."""
    resp = redirect(url_for('admin'))
    resp.delete_cookie('admin_logged_in')
    return resp

# --- API ENDPOINTS (Require Admin Authentication) ---

@app.route('/api/login', methods=['POST'])
def api_login():
    """Endpoint to set the admin login cookie."""
    data = request.get_json()
    password = data.get('password')
    if password == ADMIN_PASSWORD:
        resp = jsonify({"success": True})
        # Set a session cookie for simplicity, good for server-side auth
        resp.set_cookie('admin_logged_in', 'true', httponly=True, samesite='Lax')
        return resp, 200
    return jsonify({"success": False, "message": "Incorrect password"}), 401

@app.route('/api/licenses', methods=['GET'])
@admin_required
def get_licenses():
    """Returns the current list of licenses."""
    licenses = load_licenses()
    return jsonify({"licenses": licenses}), 200

@app.route('/api/generate', methods=['POST'])
@admin_required
def generate_license_api():
    """Generates and saves a new license."""
    data = request.get_json()
    username = data.get('username')
    hours = data.get('hours', 2)

    if not username or hours <= 0:
        return jsonify({"error": "Invalid username or hours"}), 400

    expiry_dt = datetime.utcnow() + timedelta(hours=hours)
    license = {
        "username": username,
        "password": generate_password(),
        "expiry": expiry_dt.isoformat() + 'Z' # Add Z for UTC
    }

    licenses = load_licenses()
    licenses.append(license)
    save_licenses(licenses)

    return jsonify({"success": True, "license": license}), 201

@app.route('/api/delete', methods=['POST'])
@admin_required
def delete_license_api():
    """Deletes a license at a specific index."""
    data = request.get_json()
    index = data.get('index')

    licenses = load_licenses()
    if 0 <= index < len(licenses):
        licenses.pop(index)
        save_licenses(licenses)
        return jsonify({"success": True}), 200
    
    return jsonify({"error": "Invalid license index"}), 400

@app.route('/api/clear_all', methods=['POST'])
@admin_required
def clear_all_licenses_api():
    """Clears all licenses."""
    save_licenses([])
    return jsonify({"success": True}), 200

@app.route('/api/download_file', methods=['GET'])
@admin_required
def download_licenses_file():
    """Serves the licenses.json file for download."""
    return send_file(LICENSE_FILE, as_attachment=True, download_name='licenses.json')

# --- REMOTE AUTHENTICATION ENDPOINT (For Desktop App) ---

@app.route('/api/v1/licenses/auth', methods=['POST'])
def remote_auth():
    """
    Handles remote login requests from the desktop application (SessionManager.login_remote).
    Checks credentials against licenses.json and returns a session token and expiry.
    """
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    licenses = load_licenses()

    # Find the matching license
    match = next((lic for lic in licenses if lic.get('username') == username), None)

    if not match:
        return jsonify({"error": "Username not found"}), 401
    
    # 1. Check Password
    if password != match.get('password'):
        return jsonify({"error": "Invalid password"}), 401

    # 2. Check Expiry
    expiry_str = match.get('expiry')
    if expiry_str:
        try:
            # Parse expiry string (stripping 'Z' if present for compatibility with fromisoformat)
            expiry_dt = datetime.fromisoformat(expiry_str.rstrip('Z'))
            
            # If the expiry has passed, reject the login
            if expiry_dt < datetime.utcnow():
                return jsonify({"error": "License expired"}), 403
            
            # Calculate remaining seconds for the client
            expires_in_seconds = (expiry_dt - datetime.utcnow()).total_seconds()
            
        except ValueError:
            # Handle malformed date string
            return jsonify({"error": "Invalid license expiry format"}), 500
    else:
        # Should not happen if licenses are generated correctly, but handle it
        return jsonify({"error": "License expiry date missing"}), 500

    # 3. Success: Return token and remaining expiry time
    response_data = {
        # The desktop app only needs the expiry time and a non-empty token
        "token": str(uuid.uuid4()),
        "expires_in": int(expires_in_seconds)
    }

    return jsonify(response_data), 200


if __name__ == '__main__':
    # Flask needs a 'templates' folder for render_template
    os.makedirs('templates', exist_ok=True)
    # Ensure licenses.json is initialized
    if not os.path.exists(LICENSE_FILE):
        save_licenses([])
        
    # NOTE: In a production environment like Render, the app is typically run 
    # using Gunicorn (e.g., gunicorn app:app) and this block is ignored.
    app.run(debug=True)

