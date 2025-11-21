# pore_deploy_ready_full.py
"""
Merged, deployment-ready PORE app with original features restored.

This file is a safe, cloud-deployable rewrite that restores the user's
original features from their uploaded `pore.py` while addressing the
cloud-specific issues that caused visual/layout differences.

Restored features:
- Splash screen with optional audio (gracefully disabled if missing)
- Full authentication flows: Login, Sign Up, Forgot Password (token),
  Forgot Username, Forgot Email
- DB schema upgrade helpers (safe, with WAL and retry)
- Model download from Google Drive (optional, guarded)
- Drift detection and optional retraining with versioned model saves
- ZIP download of results, plots saved to /plots
- Footer and logo rendering (graceful if asset missing)

Notes:
- Keep the models/, assets/, and Data/ folders in your repo, or enable
  model download in the UI.
- Avoid using time.sleep() that blocks reruns; where short waits are
  required we use user-friendly informative messages instead.

Place this file at the repo root and deploy to Streamlit Cloud.
"""

import os
import io
import zipfile
import logging
import sqlite3
import joblib
import time
import secrets
import string
import uuid
import base64
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import bcrypt

# ------------------------------
# Page config and basic setup
# ------------------------------
st.set_page_config(page_title="PORE Attrition Predictor", layout="wide")

BASE_PATH = Path(__file__).parent
MODEL_DIR = BASE_PATH / "models"
ASSETS_DIR = BASE_PATH / "assets"
DATA_DIR = BASE_PATH / "Data"
PLOTS_DIR = BASE_PATH / "plots"
LOG_DIR = BASE_PATH / "logs"
DB_PATH = BASE_PATH / "users.db"

for d in (MODEL_DIR, ASSETS_DIR, DATA_DIR, PLOTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Logging (rotating could be added)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "app.log")), logging.StreamHandler()],
)

# Model file paths
MODEL_PATH = MODEL_DIR / "stacked_attrition_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
COLUMNS_PATH = MODEL_DIR / "train_columns.pkl"
NUM_COLS_PATH = MODEL_DIR / "num_cols.pkl"
TRAINING_CSV = DATA_DIR / "PORE.csv"
WELCOME_AUDIO = BASE_PATH / "welcome_pore.mp3"
LOGO_PATH = ASSETS_DIR / "PORE.png"

# Google Drive file id placeholder (from original file)
DEFAULT_GDRIVE_FILE_ID = "14BxckoIrTLHYS6woIVwMYYjWSpj8JqYJ"

# ------------------------------
# DB helpers and upgrades (safe)
# ------------------------------
_db_lock = None
try:
    import threading
    _db_lock = threading.Lock()
except Exception:
    _db_lock = None

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return conn

def upgrade_users_table_safe():
    """Safely upgrade the users table schema with retry logic for DB locks."""
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

    attempts = 3
    for attempt in range(attempts):
        try:
            if _db_lock:
                _db_lock.acquire()
            with closing(get_conn()) as conn, closing(conn.cursor()) as c:
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
                table_exists = c.fetchone() is not None

                existing_cols = []
                if table_exists:
                    c.execute("PRAGMA table_info(users)")
                    existing_cols = [col[1] for col in c.fetchall()]

                # If already has desired columns, done
                if set(desired_cols.keys()).issubset(existing_cols):
                    logging.info("Users table already up-to-date")
                    return

                # Create temp table with desired schema
                temp_cols_def = ", ".join([f"{k} {v}" for k, v in desired_cols.items()])
                c.execute(f"CREATE TABLE IF NOT EXISTS users_temp ({temp_cols_def})")

                # Copy common data
                if table_exists and existing_cols:
                    common_cols = [col for col in existing_cols if col in desired_cols]
                    if common_cols:
                        cols_str = ", ".join(common_cols)
                        c.execute(f"INSERT OR IGNORE INTO users_temp ({cols_str}) SELECT {cols_str} FROM users")
                        c.execute("DROP TABLE users")

                c.execute("ALTER TABLE users_temp RENAME TO users")
                conn.commit()
                logging.info("Users table upgraded safely")
                return
        except sqlite3.OperationalError as e:
            if 'locked' in str(e).lower():
                logging.warning(f"DB locked, retrying... (attempt {attempt+1}/{attempts})")
                time.sleep(1)
                continue
            else:
                logging.exception("SQLite operational error during upgrade")
                break
        except Exception:
            logging.exception("Unexpected error during users table upgrade")
            break
        finally:
            if _db_lock and _db_lock.locked():
                try:
                    _db_lock.release()
                except Exception:
                    pass

# Add an alternate_emails column if missing
def upgrade_users_table_for_multiple_emails():
    try:
        if _db_lock:
            _db_lock.acquire()
        with closing(get_conn()) as conn, closing(conn.cursor()) as c:
            c.execute("PRAGMA table_info(users)")
            cols = [col[1] for col in c.fetchall()]
            if 'alternate_emails' not in cols:
                c.execute("ALTER TABLE users ADD COLUMN alternate_emails TEXT DEFAULT NULL")
                conn.commit()
                logging.info("Added alternate_emails column")
    except Exception:
        logging.exception("Failed to add alternate_emails column")
    finally:
        if _db_lock and _db_lock.locked():
            try:
                _db_lock.release()
            except Exception:
                pass

# Initialize DB schema safely
upgrade_users_table_safe()
upgrade_users_table_for_multiple_emails()

# ------------------------------
# Auth helpers
# ------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        if isinstance(stored_hash, str):
            stored_hash = stored_hash.encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    except Exception:
        return False


def get_user_by_email(email):
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
    if not identifier: return None
    identifier = identifier.strip().lower()
    try:
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT email FROM users WHERE LOWER(username) = ?", (identifier,))
            row = c.fetchone()
            if not row:
                c.execute("SELECT email FROM users WHERE contact = ?", (identifier,))
                row = c.fetchone()
        return row[0] if row else None
    except Exception:
        logging.exception("Error fetching email by username/contact")
        return None


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
        if c.fetchone():
            return "email_exists"
        try:
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            c.execute("INSERT INTO users (username, password, full_name, email, contact, department, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (username, hashed_pw, full_name, email, contact, department, created_at))
            conn.commit()
            return "success"
        except Exception:
            logging.exception("Signup error")
            return "error"


def login_user(username: str, password: str):
    username = username.strip()
    password = password.strip()
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT username, password, full_name, email, department FROM users WHERE username = ?", (username,))
        user = c.fetchone()
    if not user:
        return None
    stored_hash = user[1]
    try:
        if isinstance(stored_hash, str):
            stored_hash = stored_hash.encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return user
        else:
            return None
    except Exception:
        logging.exception("Error during login")
        return None


def update_password_by_email(email, new_password):
    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_pw, email))
        conn.commit()

# Reset token helpers

def create_reset_token():
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))


def store_reset_token(email, token, expiry_dt):
    expiry_str = expiry_dt.strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET reset_token = ?, reset_expiry = ? WHERE email = ?", (token, expiry_str, email))
        conn.commit()


def get_hashed_token_from_db(email):
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
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET reset_token = NULL, reset_expiry = NULL WHERE email = ?", (email,))
        conn.commit()


def verify_and_consume_token(email, token):
    stored_hashed_token, expiry_dt = get_hashed_token_from_db(email)
    if stored_hashed_token is None: return "no_token"
    if expiry_dt is None or datetime.now() > expiry_dt: return "expired"
    if not bcrypt.checkpw(token.encode('utf-8'), stored_hashed_token.encode('utf-8')):
        return "invalid_token"
    delete_token_from_db(email)
    return "ok"

# ------------------------------
# Session state defaults
# ------------------------------
for key in ["auth","username","full_name","department","splash_shown"]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ["auth","splash_shown"] else ""

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
    "fu_email": "",
}
for k,v in session_defaults.items():
    st.session_state.setdefault(k,v)

# ------------------------------
# CSS - minimal, non-invasive
# ------------------------------
st.markdown("""
<style>
.title-center {text-align:center}
.small-note {font-size:0.9rem; color: #555}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Model loading helpers
# ------------------------------
def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        logging.exception(f"Failed to load {path}: {e}")
        return None

@st.cache_resource
def load_model_resources():
    model = safe_joblib_load(MODEL_PATH) if MODEL_PATH.exists() else None
    scaler = safe_joblib_load(SCALER_PATH) if SCALER_PATH.exists() else None
    columns = safe_joblib_load(COLUMNS_PATH) if COLUMNS_PATH.exists() else None
    num_cols = safe_joblib_load(NUM_COLS_PATH) if NUM_COLS_PATH.exists() else None
    return model, scaler, columns, num_cols

model, scaler, train_columns, num_cols = load_model_resources()

# Optional: Download model from Google Drive (gdown) if missing

def download_model_from_gdrive(file_id, dest_path):
    try:
        import gdown
    except Exception:
        st.error("gdown not installed. Add it to requirements.txt to enable automatic model download.")
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(dest_path), quiet=False)
        return True
    except Exception as e:
        logging.exception("Failed to download model from Google Drive")
        return False

# ------------------------------
# UI: Authentication / Signup / Forgot flows
# ------------------------------

def show_auth_ui():
    st.markdown("<div style='background-color:#6A0DAD; padding:16px; border-radius:12px; color:white; text-align:center;'>"
                "<h2>🔒 Login / Sign Up - PORE Dashboard</h2></div>", unsafe_allow_html=True)

    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        if st.button("Login", key="nav_login"):
            st.session_state.ui_mode = "Login"
            st.experimental_rerun()
    with colB:
        if st.button("Sign Up", key="nav_signup"):
            st.session_state.ui_mode = "Sign Up"
            st.experimental_rerun()
    with colC:
        if st.button("Forgot Password", key="nav_forgot_pass"):
            st.session_state.ui_mode = "Forgot Password"
            st.experimental_rerun()
    with colD:
        if st.button("Forgot Username", key="nav_forgot_user"):
            st.session_state.ui_mode = "Forgot Username"
            st.experimental_rerun()
    with colE:
        if st.button("📧 Forgot Email?", key="btn_forgot_email"):
            st.session_state.ui_mode = "Forgot Email"
            st.experimental_rerun()

    # LOGIN
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
                    logging.exception("Error during login")
                    user = None
                if user:
                    logging.info(f"User logged in: {username}")
                    st.session_state.auth = True
                    st.session_state.username = user[0]
                    st.session_state.full_name = user[2] if len(user) > 2 else user[0]
                    st.session_state.department = user[4] if len(user) > 4 else ""
                    st.session_state.splash_shown = False
                    st.session_state.ui_mode = "splash"
                    st.experimental_rerun()
                else:
                    st.error("❌ Invalid username or password.")
        st.stop()

    # SIGN UP
    elif st.session_state.ui_mode == "Sign Up":
        st.markdown("### 📝 Create your PORE account")
        def check_password_strength(pw):
            score = 0
            if len(pw) >= 8: score += 1
            if any(c.islower() for c in pw): score += 1
            if any(c.isupper() for c in pw): score += 1
            if any(c.isdigit() for c in pw): score += 1
            if any(not c.isalnum() for c in pw): score += 1
            if score <= 2: return "Weak", "⚠️ Weak password"
            if score == 3: return "Medium", "🟡 Medium strength"
            return "Strong", "🟢 Strong password"

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
                elif pw != cpw:
                    st.error("❌ Passwords do not match.")
                else:
                    strength, msg = check_password_strength(pw)
                    if strength == "Weak":
                        st.warning(msg)
                    elif strength == "Medium":
                        st.info(msg)
                    else:
                        st.success(msg)
                    valid = True
                    if not valid:
                        st.error("Password validation failed")
                    else:
                        try:
                            result = signup_user(un, pw, fn, em, st.session_state.su_contact.strip(), st.session_state.su_department.strip())
                        except Exception:
                            logging.exception("Signup error")
                            st.error("⚠️ Something went wrong. Try again later.")
                        else:
                            if result == "success":
                                logging.info(f"New user signed up: {un} ({em})")
                                st.success("✅ Account created successfully! Please login.")
                                st.balloons()
                                st.session_state.ui_mode = "Login"
                                st.experimental_rerun()
                            elif result == "username_exists":
                                st.error("❌ Username already exists.")
                            elif result == "email_exists":
                                st.error("❌ Email already registered.")
                            else:
                                st.error("⚠️ Something went wrong.")

    # FORGOT PASSWORD
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
                    else:
                        user = get_user_by_email(em)
                        if not user:
                            st.error("❌ No account found with this Email.")
                        else:
                            token = create_reset_token()
                            expiry = datetime.now() + timedelta(minutes=15)
                            hashed_token = bcrypt.hashpw(token.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                            try:
                                store_reset_token(em, hashed_token, expiry)
                            except Exception:
                                logging.exception("Failed to store token")
                                st.error("⚠️ Could not process your request. Try again later.")
                                st.stop()
                            st.success("✅ Your reset token is generated successfully.")
                            st.info("🕐 Token valid for 15 minutes.")
                            # show token for demo (in production, email it)
                            st.code(token)
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
                    logging.info(f"Password reset for {em}")
                    st.success("✅ Password reset successfully. Login now.")
                    st.session_state.ui_mode = "Login"
                    st.experimental_rerun()
        st.stop()

    # FORGOT USERNAME
    elif st.session_state.ui_mode == "Forgot Username":
        st.markdown("### 🧾 Forgot Username")
        with st.form("forgot_username_form"):
            st.text_input("Registered Email", key="fu_email")
            submitted = st.form_submit_button("Retrieve Username")
            if submitted:
                em = st.session_state.fu_email.strip().lower()
                if not em:
                    st.error("❌ Enter a valid email.")
                else:
                    user = get_user_by_email(em)
                    if not user:
                        st.error("❌ No account found with that email.")
                    else:
                        st.success("✅ The username has been mentioned below")
                        st.info(f"Your username is: **{user[0]}**")
        st.stop()

    # FORGOT EMAIL
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
                email = get_email_by_username_or_contact(identifier)
                if email:
                    st.success(f"✅ Your registered email: **{email}**")
                else:
                    st.error("❌ No account found with that username or contact number.")
        st.stop()

# ------------------------------
# Splash screen (safe)
# ------------------------------
SPLASH_DURATION = 2.5

def show_splash_screen():
    placeholder = st.empty()
    try:
        audio_base64 = None
        if WELCOME_AUDIO.exists():
            with open(WELCOME_AUDIO, 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode()

        splash_html = f"""
        <style>
        .splash-container {{display:flex;align-items:center;justify-content:center;flex-direction:column;padding:30px;background:linear-gradient(120deg,#fff,#f3e5f5);border-radius:12px}}
        .splash-title {{color:#6A0DAD;font-size:42px}}
        .splash-sub {{color:#555;font-size:18px}}
        </style>
        <div class="splash-container">
            {f'<audio autoplay hidden><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>' if audio_base64 else ''}
            <h1 class="splash-title">✨ Welcome to <span style='color:#FF6600;'>PORE</span> ✨</h1>
            <div class="splash-sub">PREDICTIVE ORGANIZATIONAL RETENTION &amp; EFFICIENCY</div>
        </div>
        """
        placeholder.markdown(splash_html, unsafe_allow_html=True)
        time.sleep(SPLASH_DURATION)
    except Exception:
        logging.exception("Splash rendering failed")
    finally:
        st.session_state.splash_shown = True
        placeholder.empty()

# ------------------------------
# Dashboard and processing
# ------------------------------

def retrain_model(X_new, y):
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X_new, y, test_size=0.2, random_state=42)
        new_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        new_model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, new_model.predict(X_te))
        versioned_path = MODEL_DIR / f"stacked_attrition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(new_model, versioned_path)
        joblib.dump(new_model, MODEL_PATH)
        logging.info(f"Retrain success, acc={acc}")
        return True, acc
    except Exception as e:
        logging.exception("Retrain failed")
        return False, str(e)


def process_uploaded_file(uploaded_file):
    data = pd.read_csv(uploaded_file)
    st.session_state['data'] = data.copy()
    if 'employee_id' not in data.columns:
        st.error("CSV must contain 'employee_id' column.")
        return

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(data.head())

    # One-hot encode and align with train columns if available
    X_new = pd.get_dummies(data.drop(columns=['employee_id'], errors='ignore'), drop_first=True)
    if train_columns is not None:
        for col in set(train_columns) - set(X_new.columns):
            X_new[col] = 0
        X_new = X_new[train_columns]
    else:
        st.warning("Training columns not available; predictions may be unreliable.")

    # scale numeric cols if available
    if scaler is not None and num_cols is not None:
        try:
            X_new[num_cols] = scaler.transform(X_new[num_cols])
        except Exception as e:
            st.warning(f"Scaling skipped: {e}")

    # Drift detection
    uploaded_hash = None
    train_hash = None
    drifted = 0
    drift_df = None
    if TRAINING_CSV.exists():
        try:
            train_df = pd.read_csv(TRAINING_CSV)
            X_train = pd.get_dummies(train_df.drop(columns=['employee_id'], errors='ignore'), drop_first=True)
            if train_columns is not None:
                for col in set(train_columns) - set(X_train.columns):
                    X_train[col] = 0
                X_train = X_train[train_columns]
            if num_cols is not None:
                X_train[num_cols] = X_train[num_cols].fillna(0)
                X_train[num_cols] = scaler.transform(X_train[num_cols]) if scaler is not None else X_train[num_cols]

            def hash_df(df):
                col_hash = hashlib.md5("".join(df.columns).encode()).hexdigest()
                val_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
                return col_hash + val_hash

            import hashlib
            uploaded_hash = hashlib.md5(pd.util.hash_pandas_object(X_new, index=True).values).hexdigest()
            train_hash = hashlib.md5(pd.util.hash_pandas_object(X_train, index=True).values).hexdigest()

            if uploaded_hash != train_hash and num_cols is not None:
                drift_results = []
                for col in num_cols:
                    if col in X_new.columns and col in X_train.columns:
                        stat, p_value = ks_2samp(X_train[col], X_new[col])
                        drift_results.append({
                            'Feature': col,
                            'KS_Statistic': float(round(stat,4)),
                            'P_Value': float(round(p_value,4)),
                            'Drift_Detected': 'Yes' if p_value < 0.05 else 'No'
                        })
                drift_df = pd.DataFrame(drift_results)
                st.markdown('## 🔍 Data Drift Detection')
                st.dataframe(drift_df)
                drifted = drift_df[drift_df['Drift_Detected']=='Yes'].shape[0]
            else:
                st.success('✅ Uploaded data matches training data — skipping drift detection.')
        except Exception:
            logging.exception('Drift detection failed')

    # Retrain logic
    target_col = 'attrition'
    if target_col not in data.columns:
        st.warning('⚠️ Target column missing — cannot retrain.')
    else:
        if drifted > 5:
            st.warning(f'⚠️ Drift detected in {drifted} features — retraining model now...')
            ok, info = retrain_model(X_new, data[target_col])
            if ok:
                st.success(f'✅ Model retrained successfully. Test Accuracy: {info:.3f}')
                # reload model resource cache
                st.experimental_singleton.clear()
                st.experimental_rerun()
            else:
                st.error(f'❌ Retraining failed: {info}')
        elif drifted > 0:
            st.info(f'✅ Drift detected in {drifted} features, but below threshold — model not retrained.')

    # Predictions
    try:
        stack_model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        return
    if stack_model is None:
        st.error('Model file missing. Place model files into models/ or use model download option.')
        return

    pred = stack_model.predict(X_new)
    pred_prob = stack_model.predict_proba(X_new)[:,1] if hasattr(stack_model, 'predict_proba') else np.zeros(len(pred))
    results = pd.DataFrame({'employee_id': data['employee_id'], 'attrition_pred': pred, 'attrition_prob': np.round(pred_prob,4)})
    results['risk_level'] = pd.cut(results['attrition_prob'], bins=[-1,0.4,0.7,1], labels=['Low','Medium','High'])
    st.session_state['results'] = results.copy()

    st.markdown('## 📊 Employee Risk Summary')
    col1, col2, col3 = st.columns(3)
    col1.metric('🔴 High Risk', len(results[results['risk_level']=='High']))
    col2.metric('🟣 Medium Risk', len(results[results['risk_level']=='Medium']))
    col3.metric('🔵 Low Risk', len(results[results['risk_level']=='Low']))

    st.markdown('### 🧠 All Employee Risk Predictions')
    st.dataframe(results.style.applymap(lambda v: 'background-color: red; color:white' if v=='High' else ('background-color: purple; color:white' if v=='Medium' else ('background-color: blue; color:white')), subset=['risk_level']))

    # Charts
    try:
        import matplotlib.pyplot as plt
        fig_bar, ax_bar = plt.subplots(figsize=(6,4))
        results['risk_level'].value_counts().reindex(['Low','Medium','High'], fill_value=0).plot(kind='bar', ax=ax_bar)
        ax_bar.set_xlabel('Risk Level')
        ax_bar.set_ylabel('Employee Count')
        ax_bar.set_title('Employee Distribution by Risk Level')
        st.pyplot(fig_bar)

        fig_pie, ax_pie = plt.subplots(figsize=(5,5))
        risk_summary = results['risk_level'].value_counts().reindex(['Low','Medium','High'], fill_value=0)
        percentages = risk_summary / risk_summary.sum() * 100 if risk_summary.sum() else np.array([0,0,0])
        ax_pie.pie(percentages, labels=risk_summary.index, autopct='%1.1f%%')
        ax_pie.set_title('Attrition Risk Distribution')
        st.pyplot(fig_pie)

        # Save pie for ZIP
        if PLOTS_DIR.exists():
            fig_pie.savefig(PLOTS_DIR / 'risk_pie.png')
            plt.close(fig_pie)
    except Exception:
        logging.exception('Plotting failed')

# ------------------------------
# Footer rendering (safe)
# ------------------------------

def render_footer():
    try:
        logo_base64 = None
        if LOGO_PATH.exists():
            with open(LOGO_PATH, 'rb') as f:
                logo_base64 = base64.b64encode(f.read()).decode()
        footer_html = f"""
        <style>
        .footer {{text-align:center; padding:8px 0}}
        .footer img {{display:block; margin:0 auto; width:200px}}
        .footer div {{color:#555; font-size:14px}}
        </style>
        <div class='footer'>
            {f'<img src="data:image/png;base64,{logo_base64}"/>' if logo_base64 else ''}
            <div><strong>Powered by PORE</strong><br><em>Predictive Organizational Retention & Efficiency</em></div>
        </div>
        """
        st.markdown(footer_html, unsafe_allow_html=True)
    except Exception:
        logging.exception('Footer render failed')

# ------------------------------
# Main App Flow
# ------------------------------

def main():
    st.title('📊 PORE: PREDICTIVE ORGANIZATIONAL RETENTION & EFFICIENCY')
    st.markdown("<div class='small-note'>Deployment-safe version that restores original features (splash, auth, drift, retrain).</div>", unsafe_allow_html=True)

    # If not authenticated, show auth UI
    if not st.session_state.get('auth'):
        show_auth_ui()
        return

    # Splash
    if st.session_state.get('auth') and not st.session_state.get('splash_shown'):
        show_splash_screen()

    # Dashboard header
    st.markdown("""
    <div style='background:linear-gradient(90deg,#6A0DAD,#FF6600);padding:12px;border-radius:8px;color:white;text-align:center;'>
    <h2 style='margin:0'>📊 PORE Employee Attrition Prediction Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('---')

    # Sidebar
    st.sidebar.markdown(f"👨🏻‍💼 **Logged in as:** `{st.session_state.get('username','User')}`")
    if st.sidebar.button('🚪 Logout'):
        # Clear only auth-related info
        for key in ['auth','username','full_name','department']:
            st.session_state[key] = False if key=='auth' else ''
        st.session_state.ui_mode = 'Login'
        st.experimental_rerun()

    st.sidebar.markdown('---')
    st.sidebar.markdown('Model management:')
    gdrive_toggle = st.sidebar.checkbox('Attempt to download model from Google Drive if missing', value=False)
    gdrive_file_id = st.sidebar.text_input('Google Drive File ID (if using download)', value=DEFAULT_GDRIVE_FILE_ID)
    if gdrive_toggle and not MODEL_PATH.exists():
        if gdrive_file_id:
            st.sidebar.info('Attempting download...')
            ok = download_model_from_gdrive(gdrive_file_id, MODEL_PATH)
            if ok:
                st.sidebar.success('Downloaded model — reload app')
                st.experimental_rerun()
            else:
                st.sidebar.error('Download failed — check file id and gdown availability')

    st.sidebar.markdown('---')
    st.sidebar.markdown('If missing, add the following files to your repo:')
    st.sidebar.write('- models/stacked_attrition_model.pkl')
    st.sidebar.write('- models/scaler.pkl')
    st.sidebar.write('- models/train_columns.pkl')
    st.sidebar.write('- models/num_cols.pkl')
    st.sidebar.write('- Data/PORE.csv (optional, for drift checks)')
    st.sidebar.write('- assets/PORE.png (logo)')

    # File uploader
    uploaded_file = st.file_uploader('📁 Upload The Data', type=['csv'])
    if uploaded_file:
        process_uploaded_file(uploaded_file)

    # Show results / ZIP download
    if 'results' in st.session_state and isinstance(st.session_state['results'], pd.DataFrame):
        results_df = st.session_state['results']

        high_risk_df = results_df[results_df['risk_level']=='High']
        medium_risk_df = results_df[results_df['risk_level']=='Medium']

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zf:
            zf.writestr('attrition_results.csv', results_df.to_csv(index=False))
            if not high_risk_df.empty:
                zf.writestr('high_attrition_results.csv', high_risk_df.to_csv(index=False))
            if not medium_risk_df.empty:
                zf.writestr('medium_attrition_results.csv', medium_risk_df.to_csv(index=False))
            if PLOTS_DIR.exists():
                for f in os.listdir(PLOTS_DIR):
                    try:
                        zf.write(os.path.join(PLOTS_DIR, f), f)
                    except Exception:
                        pass

        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            st.download_button('💾 Download Results ZIP', data=zip_buffer.getvalue(), file_name='PORE_results.zip')
        with col2:
            if st.button('🔄 Reset / New File'):
                keep = {k:st.session_state.get(k) for k in ['auth','username','full_name','department','splash_shown','ui_mode']}
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                for k,v in keep.items():
                    st.session_state[k] = v
                if PLOTS_DIR.exists():
                    for f in os.listdir(PLOTS_DIR):
                        try:
                            os.remove(PLOTS_DIR / f)
                        except Exception:
                            pass
                st.success('🔁 Ready for a new upload!')
                st.experimental_rerun()

    # Footer
    render_footer()

if __name__ == '__main__':
    main()
