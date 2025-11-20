# ===============================================================
# 🧠 PORE Attrition Prediction Dashboard 
# ===============================================================
import secrets
import string
from email.mime.text import MIMEText
import logging
import threading
from contextlib import closing
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shutil
import io
import zipfile
import re
import secrets
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import time
from datetime import datetime, timedelta
import hashlib
import sqlite3
import bcrypt
import re
import os
import logging
from logging.handlers import RotatingFileHandler
import uuid
from datetime import datetime, timedelta
import base64
import streamlit as st

# =============== Configure Logging ===============
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "errors.log")

if not logging.getLogger().handlers:
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    console = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,  # or DEBUG for dev
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[handler, console]
    )

# =============== Page Config ===============
st.set_page_config(page_title="PORE Attrition Predictor", layout="wide")

DB_PATH = "users.db"
_db_lock = threading.Lock()

def get_conn():
    """Return a fresh SQLite connection with WAL enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def upgrade_users_table_safe():
    """Safely upgrade the users table schema with retry logic for DB locks."""
    for attempt in range(3):
        try:
            with _db_lock:
                with closing(get_conn()) as conn, closing(conn.cursor()) as c:

                    # Check if table exists
                    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
                    table_exists = c.fetchone() is not None

                    existing_cols = []
                    if table_exists:
                        c.execute("PRAGMA table_info(users)")
                        existing_cols = [col[1] for col in c.fetchall()]

                    desired_cols = {
                        "username": "TEXT PRIMARY KEY",
                        "password": "TEXT NOT NULL",
                        "full_name": "TEXT",
                        "email": "TEXT UNIQUE",
                        "contact": "TEXT",
                        "department": "TEXT",
                        "created_at": "TEXT DEFAULT (datetime('now'))",
                        "reset_token": "TEXT",
                        "reset_expiry": "TEXT"
                    }

                    # Skip if already upgraded
                    if set(desired_cols.keys()).issubset(existing_cols):
                        logging.info("✅ Users table schema already up-to-date.")
                        return

                    temp_cols_def = ", ".join([f"{k} {v}" for k, v in desired_cols.items()])
                    c.execute(f"CREATE TABLE IF NOT EXISTS users_temp ({temp_cols_def})")

                    # Copy existing data if old table present
                    if table_exists:
                        common_cols = [col for col in existing_cols if col in desired_cols]
                        if common_cols:
                            cols_str = ", ".join(common_cols)
                            c.execute(f"INSERT OR IGNORE INTO users_temp ({cols_str}) SELECT {cols_str} FROM users")
                            c.execute("DROP TABLE users")

                    c.execute("ALTER TABLE users_temp RENAME TO users")
                    conn.commit()
                    logging.info("✅ Users table upgraded safely with UNIQUE email.")
                    return  # Exit after success

        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                logging.warning(f"⚠️ Database locked, retrying in 1s... (attempt {attempt + 1}/3)")
                time.sleep(1)
                continue
            else:
                logging.exception("❌ SQLite OperationalError during upgrade")
                break
        except Exception:
            logging.exception("❌ Database upgrade failed unexpectedly")
            break
# ==================================================
# ADD COLUMN FOR MULTIPLE EMAILS
# ==================================================
def upgrade_users_table_for_multiple_emails():
    """Add a column for alternate emails if it doesn't already exist."""
    try:
        with _db_lock:
            with closing(get_conn()) as conn, closing(conn.cursor()) as c:
                # Check existing columns
                c.execute("PRAGMA table_info(users)")
                cols = [col[1] for col in c.fetchall()]

                # Add new column if missing
                if "alternate_emails" not in cols:
                    c.execute("ALTER TABLE users ADD COLUMN alternate_emails TEXT DEFAULT NULL")
                    conn.commit()
                    logging.info("✅ Added 'alternate_emails' column to users table.")
                else:
                    logging.info("ℹ️ 'alternate_emails' column already exists.")
    except Exception:
        logging.exception("❌ Failed to add 'alternate_emails' column.")
upgrade_users_table_safe()
upgrade_users_table_for_multiple_emails()

# ==================================================
# HELPERS
# ==================================================

def create_reset_token():
    """Generate a unique 8-character token (letters + numbers)."""
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))

def get_user_by_email(email):
    """Fetch user by primary or alternate email"""
    if not email:
        return None
    email = email.strip().lower()
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT * FROM users 
            WHERE LOWER(email)=? 
            OR (alternate_emails IS NOT NULL AND LOWER(alternate_emails) LIKE ?)
        """, (email, f"%{email}%"))
        return c.fetchone()

def get_email_by_username_or_contact(identifier):
    """
    Retrieve the email using username or contact number.
    Returns None if not found.
    """
    if not identifier:
        return None
    identifier = identifier.strip().lower()

    try:
        # Try by username
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT email FROM users WHERE LOWER(username) = ?", (identifier,))
            row = c.fetchone()
        # Try by contact if not found
            if not row:
                c.execute("SELECT email FROM users WHERE contact = ?", (identifier,))
                row = c.fetchone()

        return row[0] if row else None
    except Exception as e:
        logging.exception("Error while fetching email by username/contact")
        return None

def store_reset_token(email, token, expiry_dt):
    """Store hashed token and expiry in DB"""
    expiry_str = expiry_dt.strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE users SET reset_token = ?, reset_expiry = ? WHERE email = ?",
            (token, expiry_str, email)
        )
        conn.commit()

def get_hashed_token_from_db(email):
    """Return stored hashed token and expiry datetime"""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT reset_token, reset_expiry FROM users WHERE email = ?", (email,))
        row = c.fetchone()

    if not row:
        return None, None
    stored_token, expiry_str = row
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d %H:%M:%S") if expiry_str else None
    return stored_token, expiry_dt

def delete_token_from_db(email):
    """Remove reset token after successful reset"""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE users SET reset_token = NULL, reset_expiry = NULL WHERE email = ?",
            (email,)
        )
        conn.commit()

def generate_reset_token():
    """Generate a random reset token with 15-minute expiry"""
    token = str(uuid.uuid4())  # unique random token
    expiry = datetime.now() + timedelta(minutes=15)
    return token, expiry

def verify_and_consume_token(email, token):
    """Verify bcrypt token and consume it"""
    stored_hashed_token, expiry_dt = get_hashed_token_from_db(email)
    if stored_hashed_token is None: return "no_token"
    if expiry_dt is None or datetime.now() > expiry_dt: return "expired"
    if not bcrypt.checkpw(token.encode('utf-8'), stored_hashed_token.encode('utf-8')):
        return "invalid_token"
    delete_token_from_db(email)
    return "ok"

def hash_password(password: str) -> str:
    """Hash a plain text password and return a UTF-8 string"""
    hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_bytes.decode('utf-8')  # store as string

def is_valid_email(email: str) -> bool:
    return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email or "") is not None

def is_strong_password(password: str):
    if len(password) < 8:
        return False, "❌ Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "❌ Include at least one uppercase letter (e.g., A–Z)."
    if not re.search(r"[a-z]", password):
        return False, "❌ Include at least one lowercase letter (e.g., a–z)."
    if not re.search(r"[0-9]", password):
        return False, "❌ Include at least one number (e.g., 0–9)."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "❌ Include at least one special character (e.g., @, $, %, !)."
    return True, ""

def signup_user(username, password, full_name, email, contact, department):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = username.strip().lower()
    email = email.strip().lower()
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE username = ?", (username,))
        if c.fetchone():
            return "username_exists"

    c.execute("SELECT email FROM users WHERE email = ?", (email,))
    if c.fetchone(): return "email_exists"
    try:
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') 
        c.execute("""
            INSERT INTO users (username, password, full_name, email, contact, department, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username, hashed_pw, full_name, email, contact, department, created_at))
        conn.commit()
        return "success"
    except Exception as e:
        logging.exception("Signup error occurred during user registration")
        return "error"

def login_user(username: str, password: str):
    username = username.strip()
    password = password.strip()

    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT username, password, full_name, email, department FROM users WHERE username = ?",
            (username,)
        )
        user = c.fetchone()

    if not user:
        return None

    stored_hash = user[1]

    # Convert to bytes only if it's a string
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode('utf-8')

    try:
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return user
        else:
            return None
    except Exception as e:
        logging.exception("Error during login process")
        return None

def update_password_by_email(email, new_password):
    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE users SET password = ? WHERE email = ?",
            (hashed_pw, email)
        )
        conn.commit()

# ==================================================
# SESSION STATE INIT
# ==================================================
for key in ["auth","username","full_name","department","splash_shown"]:
    if key not in st.session_state: st.session_state[key] = False if key in ["auth","splash_shown"] else ""

# --- Initialize session_state safely (fixed) ---
if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = "Login"  # Always start with Login page

session_defaults = {
    "auth": False,
    "username": "",
    "full_name": "",
    "department": "",
    "splash_shown": False,
    "ui_mode": "Login", 
    "li_username": "",
    "li_password": "",
    "su_full_name": "",
    "su_email": "",
    "su_contact": "",
    "su_department": "",
    "su_username": "",
    "su_password": "",
    "su_confirm": "",
    "req_email": "",
    "reset_email": "",
    "reset_token": "",
    "reset_new": "",
    "reset_confirm": "",
    "fu_email": ""
}

for key, value in session_defaults.items():
    st.session_state.setdefault(key, value)

# ==================================================
# AUTHENTICATION UI (Login / Sign Up / Forgot)
# ==================================================
def show_auth_ui():
    st.markdown(
        "<div style='background-color:#6A0DAD; padding:20px; border-radius:12px; color:white; text-align:center;'>"
        "<h2>🔒 Login / Sign Up - PORE Dashboard</h2></div>",
        unsafe_allow_html=True
    )

    # Navigation buttons
    # ---------- Custom Button Alignment ----------
    st.markdown("""
    <style>
    div[data-testid="column"] {
        text-align: center;
    }
    button[kind="secondary"] {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        if st.button("Login", key="nav_login"):
            st.session_state.ui_mode = "Login"
            st.rerun()
    with colB:
        if st.button("Sign Up", key="nav_signup"):
            st.session_state.ui_mode = "Sign Up"
            st.rerun()
    with colC:
        if st.button("Forgot Password", key="nav_forgot_pass"):
            st.session_state.ui_mode = "Forgot Password"
            st.rerun()
    with colD:
        if st.button("Forgot Username", key="nav_forgot_user"):
            st.session_state.ui_mode = "Forgot Username"
            st.rerun()
    with colE:
        if st.button("📧 Forgot Email?", key="btn_forgot_email"):
            st.session_state.ui_mode = "Forgot Email"
            st.rerun()

    # ---------- LOGIN ----------
    if st.session_state.ui_mode == "Login":
        with st.form("login_form"):
            st.text_input("Username", key="li_username")
            st.text_input("Password", type="password", key="li_password")
            submitted = st.form_submit_button("Login")
            if submitted:
                username = st.session_state.li_username.strip()
                entered_password = st.session_state.li_password.strip()
                try:
                    user = login_user(username, entered_password)
                except Exception:
                    logging.exception("Error during user login")
                    user = None
                if user:
                    logging.info(f"User logged in: {username}")
                    st.session_state.auth = True
                    st.session_state.username = user[0]
                    st.session_state.full_name = user[2] if len(user) > 2 else user[0]
                    st.session_state.department = user[4] if len(user) > 4 else ""
                    st.session_state.splash_shown = False
                    st.session_state.ui_mode = "splash"
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
        st.stop()

    # ---------- SIGN UP ----------
    elif st.session_state.ui_mode == "Sign Up":

        st.markdown("### 📝 Create your PORE account")

        # ✅ Define password strength checker first (always accessible)
        def check_password_strength(pw):
            score = 0
            if len(pw) >= 8: score += 1
            if re.search(r"[a-z]", pw): score += 1
            if re.search(r"[A-Z]", pw): score += 1
            if re.search(r"[0-9]", pw): score += 1
            if re.search(r"[^A-Za-z0-9]", pw): score += 1

            if score <= 2:
                return "Weak", "⚠️ Weak password"
            elif score == 3:
                return "Medium", "🟡 Medium strength"
            else:
                return "Strong", "🟢 Strong password"

        # ✅ Create form
        with st.form("signup_form", clear_on_submit=True):
            st.text_input("Full Name", key="su_full_name")
            st.text_input("Email", key="su_email")
            st.text_input("Contact (optional)", key="su_contact")
            st.text_input("Department/Organization", key="su_department")
            st.text_input("Username", key="su_username")
            st.text_input("Password", type="password", key="su_password")
            st.text_input("Confirm Password", type="password", key="su_confirm")

            submitted = st.form_submit_button("Sign Up")

            if submitted:
                fn = st.session_state.su_full_name.strip()
                em = st.session_state.su_email.strip().lower()
                un = st.session_state.su_username.strip()
                pw = st.session_state.su_password.strip()
                cpw = st.session_state.su_confirm.strip()

                if not all([fn, em, un, pw, cpw]):
                    st.error("⚠️ Please fill all required fields.")
                elif not em.endswith("@gmail.com"):
                    st.error("❌ Please enter a valid Email address (must end with @gmail.com)")
                    st.stop()
                elif not is_valid_email(em):
                    st.error("❌ Invalid email format.")
                elif pw != cpw:
                    st.error("❌ Passwords do not match.")
                else:
                    # ✅ Show password strength here (after submission)
                    strength, msg = check_password_strength(pw)
                    if strength == "Weak":
                        st.warning(msg)
                    elif strength == "Medium":
                        st.info(msg)
                    else:
                        st.success(msg)

                    # ✅ Then check your strong password rule
                    valid, msg = is_strong_password(pw)
                    if not valid:
                        st.error(f"❌ {msg}")
                        st.info("💡 Example of a strong password: **A1@3$fg%6** or **Xy!9#Lp2**")
                    else:
                        try:
                            result = signup_user(
                                un, pw, fn, em,
                                st.session_state.su_contact.strip(),
                                st.session_state.su_department.strip()
                            )
                        except Exception:
                            logging.exception("Error during user signup")
                            st.error("⚠️ Something went wrong. Try again later.")
                        else:

                            if result == "success":
                                logging.info(f"New user signed up: {un} ({em})")
                                st.success("✅ Account created successfully! Please login.")
                                st.balloons()
                                st.session_state.ui_mode = "Login"
                                time.sleep(1.5)
                                st.rerun()
                            elif result == "username_exists":
                                st.error("❌ Username already exists.")
                            elif result == "email_exists":
                                st.error("❌ Email already registered.")
                            else:
                                st.error("⚠️ Something went wrong.")
    
    # ---------- FORGOT PASSWORD ----------
    elif st.session_state.ui_mode == "Forgot Password":
        st.markdown("### 🔐 Forgot Password")

        col1, col2 = st.columns(2)

    
        with col1:
            st.subheader("1️⃣ Request Reset Token")

            with st.form("request_token_form"):
                st.text_input("Registered Email", key="req_email")
                submitted = st.form_submit_button("Send reset token")

                if submitted:
                    em = st.session_state.req_email.strip().lower()
                    if not em:
                        st.error("⚠️ Please enter your registered email.")
                    elif not is_valid_email(em):
                        st.error("❌ Invalid email format.")
                    else:
                        user = get_user_by_email(em)
                        if not user:
                            st.error("❌ No account found with this Email.")
                        else:
                            token = create_reset_token()
                            st.session_state.reset_token = token  
                            expiry = datetime.now() + timedelta(minutes=15)
                            hashed_token = bcrypt.hashpw(token.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                            try:
                                store_reset_token(em, hashed_token, expiry)
                            except Exception:
                                logging.exception("Failed to store password reset token")
                                st.error("⚠️ Could not process your request. Try again later.")
                                st.stop()

                            st.success("✅ Your reset token is generated successfully.")
                            st.info("🕐 Token valid for 15 minutes.")
                            st.session_state.reset_token = token


            # ✅ Display token safely outside form
            st.markdown("### 🔑 Your Reset Token")
            if "reset_token" in st.session_state:
                st.code(st.session_state.reset_token)
                st.button("📋 Copy Token", use_container_width=True)
            else:
                st.info("Request a token first to see it here.")

        with col2, st.form("reset_password_form"):
            st.subheader("2️⃣ Reset using Token")
            st.text_input("Registered Email", key="reset_email")
            st.text_input("Reset Token", key="reset_token")
            st.text_input("New Password", type="password", key="reset_new")
            st.text_input("Confirm New Password", type="password", key="reset_confirm")
            submitted = st.form_submit_button("Reset Password")

            if submitted:
                em = st.session_state.reset_email.strip().lower()
                tok = st.session_state.reset_token.strip()
                npw = st.session_state.reset_new.strip()
                cpw = st.session_state.reset_confirm.strip()
                stored_hashed_token, expiry = get_hashed_token_from_db(em)

                if not all([em, tok, npw, cpw]):
                    st.error("⚠️ Fill all fields.")
                elif npw != cpw:
                    st.error("❌ Passwords do not match.")
                elif not stored_hashed_token:
                    st.error("❌ No reset token found.")
                elif expiry is not None and datetime.now() > expiry:
                    st.error("❌ Token expired.")
                elif not bcrypt.checkpw(tok.encode('utf-8'), stored_hashed_token.encode('utf-8')):
                    st.error("❌ Invalid token.")
                else:
                    update_password_by_email(em, npw)
                    delete_token_from_db(em)
                    safe_email = em[:3] + "***"
                    logging.info(f"Password reset successful for: {safe_email}")
                    st.success("✅ Password reset successfully. Login now.")
                    st.session_state.ui_mode = "Login"
                    time.sleep(1.5)
                    st.rerun()

        st.stop()

    # ---------- FORGOT USERNAME ----------
    elif st.session_state.ui_mode == "Forgot Username":
        st.markdown("### 🧾 Forgot Username")
        with st.form("forgot_username_form"):
            st.text_input("Registered Email", key="fu_email")
            submitted = st.form_submit_button("Retrieve Username")
            if submitted:
                em = st.session_state.fu_email.strip().lower()
                if not em or not is_valid_email(em):
                    st.error("❌ Enter a valid email.")
                else:
                    try:
                        user = get_user_by_email(em)
                    except Exception:
                        logging.exception("Failed to retrieve username by email")
                        st.error("⚠️ Could not process your request. Try again later.")
                        return

                    if not user:
                        st.error("❌ No account found with that email.")
                    else:
                        st.success("✅ The username has been mentioned below")
                        st.info(f"Your username is: **{user[0]}**")
        st.stop()

        # ---------- FORGOT EMAIL ----------
    elif st.session_state.ui_mode == "Forgot Email":
        st.markdown("### 📧 Recover Your Registered Email")

        with st.form("forgot_email_form", clear_on_submit=True):
            st.text_input("Enter your Username or Contact Number", key="fe_identifier")
            submitted = st.form_submit_button("🔍 Find My Email")

        if submitted:
            identifier = st.session_state.fe_identifier.strip() if st.session_state.fe_identifier else ""
            if not identifier:
                st.warning("⚠️ Please enter your username or contact number.")
            else:
                try:
                    email = get_email_by_username_or_contact(identifier)
                    if email:
                        st.success(f"✅ Your registered email: **{email}**")
                        st.info("💡 You can now use it to reset your password or login.")
                        safe_identifier = identifier[:2] + "***"
                        logging.info(f"Recovered email for identifier: {safe_identifier}")
                    else:
                        st.error("❌ No account found with that username or contact number.")
                except Exception:
                    logging.exception("Error occurred while recovering email")
                    st.error("⚠️ Could not process your request. Try again later.")

# ===================================
# ✅ AUTH → SPLASH → DASHBOARD 
# ===================================
SPLASH_DURATION = 3.0

# Initialize session states
if "auth" not in st.session_state:
    st.session_state.auth = False
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

# ---- Step 1: Authentication ----
if not st.session_state.get("auth"):
    show_auth_ui()
    st.stop()

# ---- Step 2: Splash -----
if st.session_state.auth and not st.session_state.splash_shown:
    import base64
    placeholder = st.empty()

    try:
        # 🎧 Encode your MP3 as Base64
        with open("welcome_pore.mp3", "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()

        # ✅ Inline HTML (no indentation!)
        splash_html = f"""
<style>
body {{
    overflow: hidden;
    background-color: #fff;
}}
@keyframes fadeInOut {{
    0% {{opacity: 0;}}
    10% {{opacity: 1;}}
    95% {{opacity: 1;}}
    100% {{opacity: 0;}}
}}
.splash-container {{
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(120deg, #fff 0%, #f3e5f5 100%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: fadeInOut 3.5s ease-in-out forwards;
    z-index: 9999;
}}
.splash-title {{
    color: #6A0DAD;
    font-size: 64px;
    font-family: 'Trebuchet MS', sans-serif;
    text-shadow: 0 0 20px #b266ff, 0 0 30px #ff6600;
}}
.splash-sub {{
    color: #555;
    font-size: 26px;
    font-family: 'Segoe UI', sans-serif;
    margin-top: 10px;
}}
</style>

<div class="splash-container">
    <audio autoplay hidden>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    <h1 class="splash-title">✨ Welcome to <span style='color:#FF6600;'>PORE</span> ✨</h1>
    <h3 class="splash-sub">PREDICTIVE ORGANIZATIONAL RETENTION & EFFICIENCY</h3>
</div>
"""

        # 👇 Render as pure HTML (not escaped Markdown)
        placeholder.markdown(splash_html, unsafe_allow_html=True)

        # Keep splash visible
        time.sleep(SPLASH_DURATION)

    except Exception:
        logging.exception("Error during splash screen rendering")
        st.warning("⚠️ There was a problem displaying the splash screen.")
    finally:
        st.session_state.splash_shown = True
        placeholder.empty()

# ---- Step 3: Dashboard ----
else:
    st.markdown("""
        <style>
        .main, [data-testid="stSidebar"], [data-testid="stHeader"] {
            animation: fadeIn 1s ease-in-out forwards;
            opacity: 0;
        }
        @keyframes fadeIn { to { opacity: 1; } }

        /* Add some spacing below the title */
        h1 {
            margin-bottom: 0.3rem;
        }
        .welcome-text {
            font-size: 1.1rem;
            color: #cfcfcf;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

st.title("📊 PORE: PREDICTIVE ORGANIZATIONAL RETENTION & EFFICIENCY")

    # 👇 This will always be visible and properly spaced
st.markdown("<div class='welcome-text'>Welcome To PORE 🚀</div>", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.markdown(f"👨🏻‍💼 **Logged in as:** `{st.session_state.get('username', 'User')}`")
if st.sidebar.button("🚪 Logout"):
    username = st.session_state.get("username", "Unknown")
    logging.info(f"User logged out: {username}")

    # Reset session state
    for key in list(st.session_state.keys()):
        if key in ["auth"]:
            st.session_state[key] = False
        elif key in ["splash_shown", "splash_start"]:
            st.session_state[key] = None 
        elif key in ["username", "full_name"]:  
            st.session_state[key] = ""
        else:
            st.session_state[key] = ""
    st.session_state.ui_mode = "Login"
    st.success("🔐Logged out successfully.")
    time.sleep(1.5)
    st.rerun()

# ---------- DELETE ACCOUNT ----------
st.sidebar.markdown("---")
st.sidebar.markdown("🗑️ **Delete Account**")
st.sidebar.write("Enter your registered email and password to confirm deletion.")

# ---- Initialize persistent states ----
st.session_state.setdefault("confirm_delete", False)
st.session_state.setdefault("account_deleted", False)
st.session_state.setdefault("ui_mode", "Dashboard")  # ✅ Default if missing

# ---- Confirmation Checkbox ----
st.session_state.confirm_delete = st.sidebar.checkbox("⚠️ Confirm Account Deletion")

# ---- Delete Account Form ----
with st.sidebar.form("delete_account_form", clear_on_submit=False):
    delete_email = st.text_input("Registered Email", key="delete_account_email")
    delete_pw = st.text_input("Password", type="password", key="delete_account_pw")

    submitted = st.form_submit_button(
        "Delete My Account",
        disabled=not st.session_state.confirm_delete
    )

    if submitted:
        username = st.session_state.get("username", "")
        if not username:
            st.sidebar.error("❌ No user logged in.")
        elif not delete_email or not delete_pw:
            st.sidebar.error("⚠️ Please enter both email and password to confirm.")
        else:
            with get_conn() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT email, password
                    FROM users
                    WHERE TRIM(LOWER(email)) = TRIM(LOWER(?))
                """, (delete_email,))
                user = c.fetchone()

            if user:
                stored_pw = user[1]
                if bcrypt.checkpw(delete_pw.encode("utf-8"), stored_pw.encode("utf-8")):
                    c.execute("""
                        DELETE FROM users
                        WHERE TRIM(LOWER(email)) = TRIM(LOWER(?))
                    """, (delete_email,))
                    conn.commit()

                    # ✅ Mark as deleted (don’t clear everything yet)
                    logging.info(f"Account deleted for: {delete_email}")
                    st.session_state.account_deleted = True
                    st.session_state.auth = False
                else:
                    st.sidebar.error("❌ Email or password incorrect.")
            else:
                st.sidebar.error("❌ Email or password incorrect.")

# ---- Handle rerun after deletion ----
if st.session_state.account_deleted:
    st.sidebar.success("✅ Account deleted successfully!")
    st.sidebar.info("Logging out...")
    time.sleep(2.0)

    # ✅ Clear only auth and user data, not core state
    for key in ["username", "auth"]:
        if key in st.session_state:
            del st.session_state[key]

    # ✅ Set UI mode to Login (safe redirect)
    st.session_state.ui_mode = "Login"
    st.session_state.account_deleted = False  # reset flag

    # ✅ Trigger a rerun safely
    st.rerun()
    
# ==================================================
# MODEL LOAD
# ==================================================

import gdown
import os
import joblib

# Paths for models
MODEL_PATH = "models/stacked_attrition_model.pkl"
SCALER_PATH = "models/scaler.pkl"
COLUMNS_PATH = "models/train_columns.pkl"
NUM_COLS_PATH = "models/num_cols.pkl"

# Google Drive file ID for the large model
file_id = "14BxckoIrTLHYS6woIVwMYYjWSpj8JqYJ"

# Create models folder if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load models
stack_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
train_columns = joblib.load(COLUMNS_PATH)
num_cols = list(joblib.load(NUM_COLS_PATH))
target_col = "attrition"

# ==================================================
# DASHBOARD HEADER
# ==================================================
st.markdown("""
<div style="background:linear-gradient(90deg,#6A0DAD,#FF6600);
padding:20px;border-radius:15px;color:white;text-align:center;font-size:30px;font-weight:bold;">
📊 PORE Employee Attrition Prediction Dashboard
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ==================================================
# FILE UPLOAD & ANALYSIS (Sequential Flow)
# ==================================================
st.markdown("<style>.splash-container{display:none!important;}</style>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload The Data", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.session_state["data"] = data.copy()  # store in session

    if 'employee_id' not in data.columns:
        st.error("❌ CSV must contain 'employee_id' column.")
    else:
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        # ==================================================
        # 🧩 One-hot encode & scale uploaded data
        # ==================================================
        X_new = pd.get_dummies(data.drop(columns=['employee_id']), drop_first=True)
        for col in set(train_columns) - set(X_new.columns):
            X_new[col] = 0
        X_new = X_new[train_columns]

        try:
            X_new[num_cols] = scaler.transform(X_new[num_cols])
        except Exception as e:
            st.warning(f"⚠️ Scaling skipped: {e}")

        # ==================================================
        # 🔍 Drift Detection
        # ==================================================
        # Load original training data
        from pathlib import Path
        BASE_PATH = Path(__file__).parent
        train_df = pd.read_csv(BASE_PATH / "Data" / "PORE.csv")

        # Function to hash dataframe
        import hashlib
        def hash_df(df):
            col_hash = hashlib.md5("".join(df.columns).encode()).hexdigest()
            val_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest() #type: ignore
            return col_hash + val_hash
        # Preprocess training data to match model columns
        X_train = pd.get_dummies(train_df.drop(columns=['employee_id', target_col]), drop_first=True)
        for col in set(train_columns) - set(X_train.columns):
            X_train[col] = 0
        X_train = X_train[train_columns]
        X_train[num_cols] = X_train[num_cols].fillna(0)
        X_train[num_cols] = scaler.transform(X_train[num_cols])

        # Hash uploaded vs training data
        uploaded_hash = hash_df(X_new)
        train_hash = hash_df(X_train)

        # Initialize drift info
        drift_df = pd.DataFrame()
        drifted = 0

        # Only perform drift detection if uploaded data differs from training
        if uploaded_hash != train_hash:
            drift_results = []
            for col in num_cols:
                if col in X_new.columns and col in X_train.columns:
                    stat, p_value = ks_2samp(X_train[col], X_new[col])
                    drift_results.append({
                        "Feature": col,
                        "KS_Statistic": float(np.round(stat, 4)), #type:ignore
                        "P_Value": float(np.round(p_value, 4)), #type:ignore
                        "Drift_Detected": "Yes" if p_value < 0.05 else "No" #type:ignore
                    })
            drift_df = pd.DataFrame(drift_results)
            st.markdown("## 🔍 Data Drift Detection")
            st.dataframe(drift_df)

            # Count drifted features
            drifted = drift_df[drift_df["Drift_Detected"] == "Yes"].shape[0]
        else:
            st.success("✅ Uploaded data matches training data — skipping drift detection.")

        # ==================================================
        # 🤖 Retrain Model if Significant Drift Detected
        # ==================================================
        def retrain_model():
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                X_new, data[target_col], test_size=0.2, random_state=42
                )
                new_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                new_model.fit(X_tr, y_tr)

                acc = accuracy_score(y_te, new_model.predict(X_te))
                versioned_path = f"{os.path.splitext(MODEL_PATH)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                joblib.dump(new_model, versioned_path)
                joblib.dump(new_model, MODEL_PATH)

                st.session_state['retrain_message'] = f"✅ Model retrained successfully! Test Accuracy: {acc:.3f}"
                st.session_state['retrain_accuracy'] = acc * 100
            except Exception as e:
                st.session_state['retrain_message'] = f"❌ Retraining failed: {e}"

        # Retrain only if enough features drifted
        if target_col not in data.columns:
            st.warning("⚠️ Target column missing — cannot retrain.")
        else:    
            if drifted > 5 and target_col in data.columns:
                st.warning(f"⚠️ Drift detected in {drifted} features — retraining model now...")
                st.session_state['retrain_message'] = "⏳ Retraining in progress..."
                retrain_model()
                st.success("✅ Retraining completed.")
            elif uploaded_hash != train_hash:
                st.success(f"✅ Drift detected in {drifted} features, but below threshold — model not retrained.")

        # ==================================================
        # 💬 Show retrain message
        # ==================================================
        if 'retrain_message' in st.session_state:
            st.info(st.session_state['retrain_message'])

        # ==================================================
        # Predictions + Risk Metrics + Tables + Charts
        # ==================================================
        # Reload model after retrain
        try:
            stack_model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load model for predictions: {e}")
            st.stop()

        pred = stack_model.predict(X_new)
        pred_prob = stack_model.predict_proba(X_new)[:,1] if hasattr(stack_model, "predict_proba") else np.zeros(len(pred))
        results = pd.DataFrame({
            "employee_id": data["employee_id"],
            "attrition_pred": pred,
            "attrition_prob": np.round(pred_prob,4)
        })

        # Risk categorization
        results["risk_level"] = pd.cut(
            results["attrition_prob"],
            bins=[-1, 0.4, 0.7, 1],
            labels=["Low", "Medium", "High"]
        )

        st.session_state["results"] = results.copy()

        # Risk Summary Metrics
        st.markdown("## 📊 Employee Risk Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 High Risk", len(results[results["risk_level"] == "High"]))
        col2.metric("🟣 Medium Risk", len(results[results["risk_level"] == "Medium"]))
        col3.metric("🔵 Low Risk", len(results[results["risk_level"] == "Low"]))

        if 'retrain_accuracy' in st.session_state:
            st.info(f"🤖 Retrain Model Accuracy: {st.session_state['retrain_accuracy']:.2f}%")

        # Styled Risk Table
        def color_risk(val):
            if val == "High": return "background-color: red; color: white;"
            elif val == "Medium": return "background-color: purple; color: white;"
            elif val == "Low": return "background-color: blue; color: white;"
            return ""

        st.markdown("### 🧠 All Employee Risk Predictions")
        st.dataframe(results.style.applymap(color_risk, subset=["risk_level"])) #type: ignore

        # Charts
        st.markdown("## 📈 Attrition Insights")
        fig_bar, ax_bar = plt.subplots(figsize=(6,4))
        results["risk_level"].value_counts().reindex(["Low","Medium","High"], fill_value=0).plot(
            kind='bar', ax=ax_bar, color=["blue","purple","red"]
        )
        ax_bar.set_xlabel("Risk Level")
        ax_bar.set_ylabel("Employee Count")
        ax_bar.set_title("Employee Distribution by Risk Level")
        st.pyplot(fig_bar)

        fig_pie, ax_pie = plt.subplots(figsize=(5,5))
        risk_summary = results["risk_level"].value_counts().reindex(["Low","Medium","High"], fill_value=0)
        percentages = risk_summary / risk_summary.sum() * 100
        ax_pie.pie(percentages, labels=risk_summary.index, autopct="%1.1f%%", colors=["#4CAF50","#FFC107","#F44336"]) #type: ignore
        ax_pie.set_title("Attrition Risk Distribution")
        st.pyplot(fig_pie)

        # Save pie for ZIP
        if os.path.exists("plots"): shutil.rmtree("plots")
        os.makedirs("plots", exist_ok=True)
        fig_pie.savefig("plots/risk_pie.png")
        plt.close(fig_pie)

# Only show ZIP download and Reset button if results exist
if 'results' in st.session_state and isinstance(st.session_state['results'], pd.DataFrame):
    zip_buffer = io.BytesIO()
    results_df = st.session_state["results"]

# Filter High & Medium risk employees
    high_risk_df = results_df[results_df["risk_level"] == "High"]
    medium_risk_df = results_df[results_df["risk_level"] == "Medium"]

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        # Full results
        zf.writestr("attrition_results.csv", results_df.to_csv(index=False))
    
        # Separate High & Medium Attrition Results
        if not high_risk_df.empty:
            zf.writestr("high_attrition_results.csv", high_risk_df.to_csv(index=False))
        if not medium_risk_df.empty:
            zf.writestr("medium_attrition_results.csv", medium_risk_df.to_csv(index=False))
    
        # Add plots
        if os.path.exists("plots"):
            for f in os.listdir("plots"):
                zf.write(os.path.join("plots", f), f)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.download_button(
            "💾 Download Results ZIP",
            data=zip_buffer.getvalue(),
            file_name="PORE_results.zip"
        )

    with col2:
        if st.button("🔄 Reset / New File", key="reset_new_file"):
            # ✅ Preserve only auth-related info
            keep = {
                "auth": st.session_state.get("auth", False),
                "username": st.session_state.get("username", ""),
                "full_name": st.session_state.get("full_name", ""),
                "department": st.session_state.get("department", ""),
                "splash_shown": st.session_state.get("splash_shown", True),
                "ui_mode": st.session_state.get("ui_mode", "Dashboard"),
            }

            # ✅ Clear everything else
            for key in list(st.session_state.keys()):
                del st.session_state[key]

            # ✅ Restore preserved info
            for k, v in keep.items():
                st.session_state[k] = v

            # ✅ Clear plots if they exist
            if os.path.exists("plots"):
                for f in os.listdir("plots"):
                    os.remove(os.path.join("plots", f))

            st.success("🔁 Ready for a new upload!")
            time.sleep(1)
            st.rerun()

# ==================================================
# 🌟 Footer / Powered by PORE (Centered, Natural Flow)
# ==================================================

st.markdown("---")

# Path to logo
logo_path = os.path.join("assets", "PORE.png")

# Convert image to base64
with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

# Centered footer without fixed positioning
footer_html = f"""
<style>
.footer {{
    text-align: center;
    background-color: transparent;
    padding: 10px 0;
}}

.footer img {{
    display: block;
    margin: 0 auto;
    width: 250px; 
    opacity: 0.9;
}}

.footer div {{
    color: #fff;
    font-size: 18px;
    text-align: center;
    line-height: 1.3;
}}
</style>

<div class="footer">
    <img src="data:image/png;base64,{logo_base64}" alt="PORE Logo">
    <div>
        <strong>Powered by PORE</strong><br>
        <em>Predictive Organizational Retention & Efficiency</em>
    </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)