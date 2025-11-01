# -*- coding: utf-8 -*-
# app.py — Arxora Trading Platform v4.0 (Production Ready)

import os
import re
import sys
import importlib
import traceback
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import streamlit as st

# ========= Model Loading =========
try:
    from core.model_fetch import ensure_models
    try:
        ensure_models()
    except Exception as _e:
        import logging as _lg
        _lg.warning("ensure_models failed: %s", _e)
except Exception as _e:
    import logging as _lg
    _lg.warning("model_fetch import skipped: %s", _e)

# ========= Database =========
try:
    from database import TradingDatabase
    db = TradingDatabase()
except Exception as e:
    st.error(f"Database initialization failed: {e}")
    st.stop()

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import requests
except Exception:
    requests = None

# ========= Environment =========
MODEL_DIR = Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
ARXORA_DEBUG = os.getenv("ARXORA_DEBUG", "0") == "1"
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT",  "0.0010"))

# ========= Page Config =========
st.set_page_config(
    page_title="Arxora",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========= Professional Theme =========
st.markdown("""
<style>
:root {
    --bg-primary: #000000;
    --bg-secondary: #0a0a0a;
    --bg-tertiary: #141414;
    --surface: #1a1a1a;
    --surface-elevated: #202020;
    --surface-hover: #252525;
    --accent-primary: #16c784;
    --accent-blue: #5B7FF9;
    --success: #16c784;
    --danger: #ea3943;
    --warning: #ffa94d;
    --text-primary: #ffffff;
    --text-secondary: #a0a0a0;
    --text-tertiary: #707070;
    --text-disabled: #404040;
    --border: rgba(255, 255, 255, 0.1);
    --border-light: rgba(255, 255, 255, 0.05);
    --sidebar-width: 300px;
}

* {
    font-feature-settings: "tnum" 1;
    -webkit-font-smoothing: antialiased;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif !important;
}

#MainMenu, footer, header, [data-testid="stHeader"] {
    visibility: hidden !important;
    display: none !important;
}

.stDeployButton {display: none !important;}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

::-webkit-scrollbar {width: 6px; height: 6px;}
::-webkit-scrollbar-track {background: var(--bg-secondary);}
::-webkit-scrollbar-thumb {background: var(--text-disabled); border-radius: 3px;}

/* Sidebar */
.account-sidebar {
    width: var(--sidebar-width);
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    padding: 2rem 1.5rem;
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;
    overflow-y: auto;
    z-index: 1000;
}

.sidebar-header {margin-bottom: 2rem;}
.sidebar-logo {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}

.sidebar-subtitle {
    font-size: 11px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

.account-info {
    background: var(--surface);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-light);
}

.account-label {
    font-size: 11px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.account-value {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

.account-change {font-size: 14px; font-weight: 600;}
.account-change.positive {color: var(--success);}
.account-change.negative {color: var(--danger);}

.stats-container {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-item {
    background: var(--surface);
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid var(--border-light);
}

.stat-label {
    font-size: 10px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
}

.sidebar-section {
    margin: 2rem 0;
    padding-top: 2rem;
    border-top: 1px solid var(--border-light);
}

/* Main Content */
.main-content {
    margin-left: var(--sidebar-width);
    width: calc(100% - var(--sidebar-width));
    padding: 2rem 3rem;
    min-height: 100vh;
}

.main-content.auth-mode {
    margin-left: 0;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

@media (max-width: 1024px) {
    .account-sidebar {
        transform: translateX(-100%);
        transition: transform 0.3s ease;
    }
    .account-sidebar.mobile-open {
        transform: translateX(0);
    }
    .main-content {
        margin-left: 0;
        width: 100%;
        padding: 1.5rem;
    }
}

/* Mobile Menu Toggle */
.mobile-menu-btn {
    position: fixed;
    top: 1rem;
    left: 1rem;
    z-index: 1001;
    width: 40px;
    height: 40px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    display: none;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    color: var(--text-primary);
}

@media (max-width: 1024px) {
    .mobile-menu-btn {
        display: flex;
    }
}

/* Auth Page */
.auth-container {
    max-width: 480px;
    width: 100%;
    padding: 2rem;
}

.auth-header {
    text-align: center;
    margin-bottom: 3rem;
}

.auth-logo {
    font-size: 36px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}

.auth-subtitle {
    font-size: 11px;
    color: #707070;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Signal Display */
.signal-card {
    background: var(--bg-secondary);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid var(--border);
}

.signal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem 1.5rem;
    background: linear-gradient(90deg, rgba(22, 199, 132, 0.2), rgba(22, 199, 132, 0.05));
    border-radius: 12px;
    margin-bottom: 2rem;
    border: 1px solid rgba(22, 199, 132, 0.3);
}

.signal-header.short {
    background: linear-gradient(90deg, rgba(234, 57, 67, 0.2), rgba(234, 57, 67, 0.05));
    border-color: rgba(234, 57, 67, 0.3);
}

.signal-header.wait {
    background: linear-gradient(90deg, rgba(112, 112, 112, 0.2), rgba(112, 112, 112, 0.05));
    border-color: rgba(112, 112, 112, 0.3);
}

.signal-title {display: flex; align-items: center; gap: 1rem;}

.signal-badge {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.signal-badge.long {background: var(--success); color: #000;}
.signal-badge.short {background: var(--danger); color: #fff;}
.signal-badge.wait {background: var(--text-disabled); color: var(--text-secondary);}

.signal-name {font-size: 18px; font-weight: 600; color: var(--text-primary);}
.signal-confidence {font-size: 16px; font-weight: 600; color: var(--text-secondary);}

.asset-display {text-align: center; margin: 2rem 0;}
.asset-name {font-size: 18px; font-weight: 600; color: var(--text-secondary); margin-bottom: 1rem;}
.asset-price {font-size: 56px; font-weight: 700; color: var(--text-primary); letter-spacing: -2px;}

/* Levels Grid */
.levels-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.25rem;
    margin: 2rem 0;
}

.level-card {
    background: var(--surface);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--border);
    transition: all 0.2s;
}

.level-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
}

.level-card.entry {
    border-color: var(--success);
    background: linear-gradient(135deg, rgba(22, 199, 132, 0.1), rgba(22, 199, 132, 0.02));
}

.level-card.stoploss {
    border-color: var(--danger);
    background: linear-gradient(135deg, rgba(234, 57, 67, 0.1), rgba(234, 57, 67, 0.02));
}

.level-label {
    font-size: 11px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 0.875rem;
}

.level-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.level-detail {font-size: 13px; color: var(--text-secondary);}

/* Confidence Meter */
.confidence-meter {
    background: var(--bg-tertiary);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 2rem 0;
    border: 1px solid var(--border-light);
}

.confidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.confidence-label {
    font-size: 13px;
    color: var(--text-tertiary);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.confidence-value {font-size: 24px; font-weight: 700; color: var(--text-primary);}

.confidence-bar {
    height: 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1rem;
}

.confidence-fill {
    height: 100%;
    background: var(--success);
    transition: width 0.6s ease;
}

.confidence-info {
    font-size: 12px;
    color: var(--text-tertiary);
    font-family: "SF Mono", monospace;
}

/* Inputs */
.stTextInput input, .stNumberInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 15px !important;
    padding: 0.875rem !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent-primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(22, 199, 132, 0.1) !important;
}

.stTextInput label, .stNumberInput label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    padding: 1rem 2rem !important;
    background: var(--accent-primary) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #14b578 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(22, 199, 132, 0.3) !important;
}

.stButton > button[kind="primary"] {
    background: var(--accent-blue) !important;
    color: #fff !important;
}

.stButton > button[kind="primary"]:hover {
    background: #4a6df0 !important;
    box-shadow: 0 4px 12px rgba(91, 127, 249, 0.3) !important;
}

/* Radio Buttons */
.stRadio > div {display: flex; gap: 1rem; flex-wrap: wrap;}

.stRadio > div > label {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.875rem 1.5rem !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

.stRadio > div > label:hover {
    border-color: var(--text-secondary) !important;
    background: var(--surface-hover) !important;
}

.stRadio > div > label[data-checked="true"] {
    background: var(--accent-primary) !important;
    border-color: var(--accent-primary) !important;
    color: #000 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-tertiary) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 1rem 0 !important;
    border: none !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
}

/* Alerts */
.stAlert {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    border-left-width: 3px !important;
    color: var(--text-primary) !important;
}

.stInfo {border-left-color: var(--accent-blue) !important;}
.stWarning {border-left-color: var(--warning) !important;}
.stError {border-left-color: var(--danger) !important;}
.stSuccess {border-left-color: var(--success) !important;}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
}

[data-testid="stMetric"] label {
    font-size: 11px !important;
    color: var(--text-tertiary) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* DataFrames */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

.stDataFrame table {background: var(--surface) !important;}

.stDataFrame thead tr th {
    background: var(--bg-tertiary) !important;
    color: var(--text-tertiary) !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 1rem !important;
    border-bottom: 1px solid var(--border) !important;
}

.stDataFrame tbody tr td {
    padding: 1rem !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-light) !important;
}

.stDataFrame tbody tr:hover {background: var(--surface-hover) !important;}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    padding: 1rem !important;
}

.streamlit-expanderHeader:hover {
    background: var(--surface-hover) !important;
    border-color: var(--border) !important;
}

.streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 1.25rem !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    background: var(--surface);
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent-primary) !important;
    border: 2px solid var(--accent-primary) !important;
}

@media (max-width: 768px) {
    .levels-grid {grid-template-columns: 1fr;}
    .stats-container {grid-template-columns: 1fr;}
    .asset-price {font-size: 40px;}
    .signal-header {flex-direction: column; gap: 1rem; text-align: center;}
}

</style>
""", unsafe_allow_html=True)

# ========= Helper Functions =========
def _user_exists_in_current_db(username: str) -> bool:
    """Check if user exists in database"""
    name = (username or "").strip()
    if not name:
        return False
    conn = None
    try:
        conn = sqlite3.connect(db.db_name, timeout=10)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (name,))
        return cur.fetchone() is not None
    except Exception:
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "0.00"

def sanitize_targets(action: str, entry: float, tp1: float, tp2: float, tp3: float):
    step = max(MIN_TP_STEP_PCT * max(1.0, abs(entry)), 1e-6 * max(1.0, abs(entry)))
    if action == "BUY":
        a = sorted([tp1, tp2, tp3])
        a[0] = max(a[0], entry + step)
        a[1] = max(a[1], a[0] + step)
        a[2] = max(a[2], a[1] + step)
        return a[0], a[1], a[2]
    if action == "SHORT":
        a = sorted([tp1, tp2, tp3], reverse=True)
        a[0] = min(a[0], entry - step)
        a[1] = min(a[1], a[0] - step)
        a[2] = min(a[2], a[1] - step)
        return a[0], a[1], a[2]
    return tp1, tp2, tp3

def entry_mode_labels(action: str, entry: float, last_price: float, eps: float):
    if action not in ("BUY", "SHORT"):
        return "WAIT", "Entry"
    if abs(entry - last_price) <= eps * max(1.0, abs(last_price)):
        return "Market", "Entry (Market)"
    if action == "BUY":
        return ("Buy Stop", "Entry (Buy Stop)") if entry > last_price else ("Buy Limit", "Entry (Buy Limit)")
    else:
        return ("Sell Stop", "Entry (Sell Stop)") if entry < last_price else ("Sell Limit", "Entry (Sell Limit)")

def normalize_for_polygon(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace(" ", "")
    if ":" in s:
        head, tail = s.split(":", 1)
        tail = tail.replace("USDT", "USD").replace("USDC", "USD")
        return f"{head}:{tail}"
    if re.match(r"^[A-Z]{2,10}USD(T|C)?$", s or ""):
        s = s.replace("USDT", "USD").replace("USDC", "USD")
        return f"X:{s}"
    return s

def rr_line(levels: Dict[str, float]) -> str:
    risk = abs(levels["entry"] - levels["sl"])
    if risk <= 1e-9:
        return ""
    rr1 = abs(levels["tp1"] - levels["entry"]) / risk
    rr2 = abs(levels["tp2"] - levels["entry"]) / risk
    rr3 = abs(levels["tp3"] - levels["entry"]) / risk
    return f"1:{rr1:.1f} (TP1) · 1:{rr2:.1f} (TP2) · 1:{rr3:.1f} (TP3)"

@st.cache_data(show_spinner=False, ttl=86400)
def resolve_asset_title_polygon(raw_symbol: str, normalized: str) -> str:
    s = (raw_symbol or "").strip().upper()
    t = (normalized or s).strip().upper()
    api = os.getenv("POLYGON_API_KEY") or os.getenv("POLYGON_KEY")
    if not api or requests is None:
        return s
    try:
        r = requests.get(
            f"https://api.polygon.io/v3/reference/tickers/{t}",
            params={"apiKey": api},
            timeout=2.5
        )
        if r.ok:
            data = r.json() or {}
            name = ((data.get("results") or {}).get("name") or "").strip()
            if name:
                return f"{name} ({s})"
    except Exception:
        pass
    return s

# ========= Cached Data Functions =========
@st.cache_data(ttl=60, show_spinner=False)
def get_cached_user_info(user_id: int):
    """Cache user info for 60 seconds"""
    try:
        return db.get_user_info(user_id)
    except Exception as e:
        if ARXORA_DEBUG:
            st.error(f"Failed to load user info: {e}")
        return None

@st.cache_data(ttl=60, show_spinner=False)
def get_cached_stats(user_id: int):
    """Cache stats for 60 seconds"""
    try:
        return db.get_statistics(user_id)
    except Exception as e:
        if ARXORA_DEBUG:
            st.error(f"Failed to load stats: {e}")
        return {
            'total_trades': 0,
            'closed_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0
        }

# ========= Strategy Loading =========
try:
    import services.data  # noqa
except Exception:
    try:
        import core.data as _core_data
        sys.modules['services.data'] = _core_data
    except Exception:
        pass

def _load_strategy_module():
    try:
        mod = importlib.import_module("core.strategy")
        try:
            mod = importlib.reload(mod)
        except Exception:
            pass
        return mod, None
    except Exception:
        return None, traceback.format_exc()

def get_available_models() -> List[str]:
    mod, _ = _load_strategy_module()
    if not mod:
        return ["Octopus"]
    reg = getattr(mod, "STRATEGY_REGISTRY", {}) or {}
    keys = list(reg.keys())
    return (["Octopus"] if "Octopus" in keys else []) + [k for k in sorted(keys) if k != "Octopus"]

def run_model_by_name(ticker_norm: str, model_name: str) -> Dict[str, Any]:
    mod, err = _load_strategy_module()
    if not mod:
        raise RuntimeError("Failed to import core.strategy:\n" + (err or ""))
    if hasattr(mod, "analyze_asset"):
        return mod.analyze_asset(ticker_norm, "Краткосрочный", model_name)
    reg = getattr(mod, "STRATEGY_REGISTRY", {}) or {}
    if model_name in reg and callable(reg[model_name]):
        return reg[model_name](ticker_norm, "Краткосрочный")
    fname = f"analyze_asset_{model_name.lower()}"
    if hasattr(mod, fname):
        return getattr(mod, fname)(ticker_norm, "Краткосрочный")
    raise RuntimeError(f"Strategy {model_name} is not available.")

# ========= Performance Tracking =========
try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# ========= Authentication Page =========
def show_auth_page():
    """Clean auth page without sidebar or extra elements"""
    
    # Auth mode styling
    st.markdown('<div class="main-content auth-mode">', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="auth-container">
            <div class="auth-header">
                <div class="auth-logo">Arxora</div>
                <div class="auth-subtitle">AI-Powered Trading Intelligence</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.markdown("### Sign In")
        
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        if st.button("Sign In", type="primary", use_container_width=True):
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                try:
                    user = db.login_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                        if username and not _user_exists_in_current_db(username):
                            st.info("User not found. Please register first.")
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
                    if ARXORA_DEBUG:
                        st.exception(e)
    
    with tab2:
        st.markdown("### Create Account")
        
        new_username = st.text_input("Username", key="reg_username", placeholder="Choose a username (min 3 characters)")
        new_password = st.text_input("Password", type="password", key="reg_password", placeholder="Choose a password (min 6 characters)")
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000, help="Virtual starting capital for trading simulation")
        
        if st.button("Create Account", type="primary", use_container_width=True):
            # Validation
            if len((new_username or "").strip()) < 3:
                st.error("Username must be at least 3 characters")
            elif len((new_password or "").strip()) < 6:
                st.error("Password must be at least 6 characters")
            else:
                try:
                    # Check if user already exists BEFORE attempting to register
                    if _user_exists_in_current_db(new_username):
                        st.error("Username already taken. Please choose another one.")
                    else:
                        # Attempt registration
                        user_id = db.register_user(new_username, new_password, initial_capital)
                        if user_id:
                            # Auto-login after successful registration
                            user = db.login_user(new_username, new_password)
                            if user:
                                st.session_state.user = user
                                st.success("Account created successfully! Logging you in...")
                                st.balloons()
                                st.rerun()
                            else:
                                st.success("Account created! Please sign in.")
                        else:
                            st.error("Registration failed. Username might be taken.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists. Please choose a different one.")
                except Exception as e:
                    st.error(f"Registration failed: {str(e)}")
                    if ARXORA_DEBUG:
                        st.exception(e)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========= AUTHENTICATION CHECK - MUST BE FIRST =========
if 'user' not in st.session_state:
    show_auth_page()
    st.stop()

# ========= Initialize Session State =========
if 'min_confidence_filter' not in st.session_state:
    st.session_state['min_confidence_filter'] = 60

if 'mobile_sidebar_open' not in st.session_state:
    st.session_state['mobile_sidebar_open'] = False

# ========= Get User Data (Cached) =========
user_info = get_cached_user_info(st.session_state.user['user_id'])
stats = get_cached_stats(st.session_state.user['user_id'])

# Safety check
if not user_info:
    st.error("Failed to load user data. Please try logging in again.")
    if st.button("Logout and Retry"):
        del st.session_state.user
        st.rerun()
    st.stop()

# ========= Render Sidebar (ONLY AFTER AUTH) =========
def render_sidebar():
    """Render sidebar with user information"""
    pnl_change = user_info['current_capital'] - user_info['initial_capital']
    pnl_percent = (pnl_change / max(1e-9, user_info['initial_capital'])) * 100
    pnl_class = "positive" if pnl_change >= 0 else "negative"
    pnl_sign = "+" if pnl_change >= 0 else ""
    
    mobile_class = "mobile-open" if st.session_state.get('mobile_sidebar_open', False) else ""
    
    st.markdown(f'''
    <div class="account-sidebar {mobile_class}">
        <div class="sidebar-header">
            <div class="sidebar-logo">Arxora</div>
            <div class="sidebar-subtitle">Trading Intelligence</div>
        </div>
        
        <div class="account-info">
            <div class="account-label">Current Capital</div>
            <div class="account-value">${user_info['current_capital']:,.2f}</div>
            <div class="account-change {pnl_class}">
                {pnl_sign}${abs(pnl_change):,.2f} ({pnl_sign}{pnl_percent:.2f}%)
            </div>
        </div>
        
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{stats['total_trades']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value">{stats['win_rate']:.1f}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Closed</div>
                <div class="stat-value">{stats['closed_trades']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg P&L</div>
                <div class="stat-value">{stats['avg_pnl']:.2f}%</div>
            </div>
        </div>
        
        <div class="sidebar-section">
            <div style="background: var(--surface); padding: 1rem; border-radius: 8px; border: 1px solid var(--border-light);">
                <div style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Min. Confidence</div>
                <div style="font-size: 20px; font-weight: 700; color: var(--text-primary);">{st.session_state.get('min_confidence_filter', 60)}%</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

render_sidebar()

# ========= Main Content =========
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Mobile menu toggle button
if st.button("☰", key="mobile_menu_toggle"):
    st.session_state['mobile_sidebar_open'] = not st.session_state.get('mobile_sidebar_open', False)
    st.rerun()

# ========= Tabs =========
tab_signals, tab_portfolio, tab_active, tab_stats = st.tabs([
    "AI Signals",
    "Portfolio",
    "Active Trades",
    "Statistics"
])

# ========= TAB 1: AI Signals =========
with tab_signals:
    st.markdown("### Trading Agent Analysis")
    
    models = get_available_models()
    model = st.radio("Select AI Model", options=models, index=0, horizontal=True, key="agent_radio")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input("Asset Symbol", placeholder="AAPL, SPY, BTCUSD, C:EURUSD", help="Enter stock ticker, crypto pair, or forex symbol")
    with col2:
        min_conf = st.number_input("Min. Confidence %", min_value=0, max_value=100, value=st.session_state.get('min_confidence_filter', 60), step=5)
        st.session_state['min_confidence_filter'] = min_conf
    
    ticker = ticker_input.strip().upper()
    symbol_for_engine = normalize_for_polygon(ticker)
    
    if st.button("Analyze Asset", type="primary", use_container_width=True):
        if not ticker:
            st.warning("Please enter an asset symbol")
        else:
            with st.spinner(f"Analyzing {ticker} with {model}..."):
                try:
                    output = run_model_by_name(symbol_for_engine, model)
                    
                    rec = output.get("recommendation")
                    if not rec and ("action" in output or "confidence" in output):
                        rec = {"action": output.get("action", "WAIT"), "confidence": float(output.get("confidence", 0.0))}
                    if not rec:
                        rec = {"action": "WAIT", "confidence": 0.0}
                    
                    action = str(rec.get("action", "WAIT"))
                    conf_val = float(rec.get("confidence", 0.0))
                    conf_pct_val = conf_val * 100.0 if conf_val <= 1.0 else conf_val
                    
                    last_price = float(output.get("last_price", 0.0) or 0.0)
                    
                    lv = {k: float(output.get("levels", {}).get(k, 0.0)) for k in ("entry", "sl", "tp1", "tp2", "tp3")}
                    
                    if action in ("BUY", "SHORT"):
                        tp1, tp2, tp3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
                        lv["tp1"], lv["tp2"], lv["tp3"] = float(tp1), float(tp2), float(tp3)
                    
                    st.session_state["last_signal"] = {
                        "ticker": ticker,
                        "symbol_for_engine": symbol_for_engine,
                        "action": action,
                        "confidence": conf_pct_val,
                        "model": model,
                        "output": output,
                        "last_price": last_price,
                        "levels": lv
                    }
                    
                    asset_title = resolve_asset_title_polygon(ticker, symbol_for_engine)
                    direction_class = "long" if action == "BUY" else ("short" if action == "SHORT" else "wait")
                    
                    st.markdown('<div class="signal-card">', unsafe_allow_html=True)
                    
                    mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
                    direction_text = f"{'LONG' if action == 'BUY' else ('SHORT' if action == 'SHORT' else 'WAIT')} — {mode_text}"
                    
                    st.markdown(f'''
                        <div class="signal-header {direction_class}">
                            <div class="signal-title">
                                <div class="signal-badge {direction_class}">
                                    {"BUY" if action == "BUY" else ("SELL" if action == "SHORT" else "WAIT")}
                                </div>
                                <div class="signal-name">{direction_text}</div>
                            </div>
                            <div class="signal-confidence">Confidence: {int(round(conf_pct_val))}%</div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown(f'''
                        <div class="asset-display">
                            <div class="asset-name">{asset_title}</div>
                            <div class="asset-price">${last_price:.2f}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    now_utc = datetime.now(timezone.utc)
                    st.caption(f"Analysis: {now_utc.strftime('%Y-%m-%d %H:%M UTC')} · Model: {model}")
                    
                    ai_delta = conf_pct_val - float(st.session_state.get("last_rules_pct", 44.0))
                    st.markdown(f'''
                        <div class="confidence-meter">
                            <div class="confidence-header">
                                <div class="confidence-label">Confidence Level</div>
                                <div class="confidence-value">{conf_pct_val:.0f}%</div>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {conf_pct_val}%"></div>
                            </div>
                            <div class="confidence-info">
                                AI Override: {ai_delta:+.0f}% · Combined algorithmic and ML analysis
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    if action in ("BUY", "SHORT"):
                        probs = output.get('probs', {})
                        
                        st.markdown('<div class="levels-grid">', unsafe_allow_html=True)
                        
                        st.markdown(f'''
                            <div class="level-card entry">
                                <div class="level-label">{entry_title}</div>
                                <div class="level-value">${lv["entry"]:.2f}</div>
                            </div>
                            <div class="level-card stoploss">
                                <div class="level-label">Stop Loss</div>
                                <div class="level-value">${lv["sl"]:.2f}</div>
                            </div>
                            <div class="level-card">
                                <div class="level-label">Take Profit 1</div>
                                <div class="level-value">${lv["tp1"]:.2f}</div>
                                <div class="level-detail">30% · {int(round(probs.get('tp1', 0)*100))}% prob</div>
                            </div>
                            <div class="level-card">
                                <div class="level-label">Take Profit 2</div>
                                <div class="level-value">${lv["tp2"]:.2f}</div>
                                <div class="level-detail">30% · {int(round(probs.get('tp2', 0)*100))}% prob</div>
                            </div>
                            <div class="level-card">
                                <div class="level-label">Take Profit 3</div>
                                <div class="level-value">${lv["tp3"]:.2f}</div>
                                <div class="level-detail">40% · {int(round(probs.get('tp3', 0)*100))}% prob</div>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        rr = rr_line(lv)
                        if rr:
                            st.info(f"Risk/Reward: {rr}")
                    
                    st.caption("AI analysis is for informational purposes only and does not constitute investment advice.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    try:
                        log_agent_performance(model, ticker, datetime.today(), 0.0)
                    except Exception:
                        pass
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    if ARXORA_DEBUG:
                        st.exception(e)
    else:
        st.info("Enter an asset symbol and click Analyze to generate AI-powered trading signals.")

# ========= TAB 2: Portfolio =========
with tab_portfolio:
    st.markdown("### Add Signal to Portfolio")
    
    if "last_signal" in st.session_state:
        sig = st.session_state["last_signal"]
        ticker = sig["ticker"]
        action = sig["action"]
        conf = sig["confidence"]
        lv = sig["levels"]
        
        if action not in ("BUY", "SHORT"):
            st.warning("Last signal was WAIT. Cannot add to portfolio.")
        elif not db.can_add_trade(st.session_state.user['user_id'], ticker):
            st.warning(f"Active trade already exists for {ticker}. Close it before adding a new one.")
        else:
            st.success(f"Signal: **{ticker}** — **{action}** ({conf:.0f}% confidence)")
            
            st.markdown("#### Trade Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry", f"${lv['entry']:.2f}")
            with col2:
                st.metric("Stop Loss", f"${lv['sl']:.2f}")
            with col3:
                risk_pct = abs(lv['entry'] - lv['sl']) / max(1e-9, abs(lv['entry'])) * 100
                st.metric("Risk %", f"{risk_pct:.2f}%")
            
            st.markdown("#### Position Sizing")
            position_percent = st.slider("Portfolio Allocation (%)", min_value=5, max_value=50, value=10, step=5, help="Percentage of your capital to allocate")
            
            position_size = (user_info['current_capital'] * position_percent) / 100
            st.info(f"Position Size: **${position_size:,.2f}** ({position_percent}% of capital)")
            
            st.markdown("#### Profit/Loss Projections")
            potential_profit = position_size * abs(lv['tp1'] - lv['entry']) / max(1e-9, abs(lv['entry']))
            potential_loss = position_size * abs(lv['entry'] - lv['sl']) / max(1e-9, abs(lv['entry']))
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Potential Profit (TP1): **${potential_profit:.2f}**")
            with col2:
                st.error(f"Potential Loss (SL): **${potential_loss:.2f}**")
            
            if conf < st.session_state['min_confidence_filter']:
                st.warning(f"⚠️ Signal confidence ({conf:.0f}%) is below your threshold ({st.session_state['min_confidence_filter']}%). Consider waiting for a stronger signal.")
            
            if st.button("Add to Portfolio", type="primary", use_container_width=True):
                try:
                    signal_data = {
                        'ticker': ticker,
                        'direction': 'LONG' if action == 'BUY' else 'SHORT',
                        'entry_price': lv['entry'],
                        'stop_loss': lv['sl'],
                        'tp1': lv['tp1'],
                        'tp2': lv['tp2'],
                        'tp3': lv['tp3'],
                        'tp1_prob': float(sig['output'].get('probs', {}).get('tp1', 0) * 100),
                        'tp2_prob': float(sig['output'].get('probs', {}).get('tp2', 0) * 100),
                        'tp3_prob': float(sig['output'].get('probs', {}).get('tp3', 0) * 100),
                        'confidence': int(conf),
                        'model': sig['model']
                    }
                    trade_id = db.add_trade(st.session_state.user['user_id'], signal_data, position_percent)
                    st.success(f"Trade added successfully! Trade ID: #{trade_id}")
                    st.balloons()
                    # Clear cache
                    get_cached_user_info.clear()
                    get_cached_stats.clear()
                    del st.session_state["last_signal"]
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to add trade: {str(e)}")
                    if ARXORA_DEBUG:
                        st.exception(e)
    else:
        st.info("No signal available. Analyze an asset in the **AI Signals** tab first.")

# ========= TAB 3: Active Trades =========
with tab_active:
    st.markdown("### Active Trades")
    
    active_trades = db.get_active_trades(st.session_state.user['user_id'])
    
    if not active_trades:
        st.info("No active trades. Add a signal from the **Portfolio** tab.")
    else:
        for trade in active_trades:
            sl_status = "Breakeven" if trade['sl_breakeven'] else f"${trade['stop_loss']:.2f}"
            
            with st.expander(f"{trade['ticker']} — {trade['direction']} | {trade['remaining_percent']:.0f}% remaining | SL: {sl_status}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entry", f"${trade['entry_price']:.2f}")
                with col2:
                    st.metric("Position", f"${trade['position_size']:.2f}")
                with col3:
                    st.metric("Model", trade['model_used'])
                with col4:
                    st.metric("Confidence", f"{trade['confidence']}%")
                
                st.markdown("**Take Profit Progress**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    status = "✓" if trade['tp1_closed'] else "○"
                    st.markdown(f"{status} TP1 (30%)")
                with col2:
                    status = "✓" if trade['tp2_closed'] else "○"
                    st.markdown(f"{status} TP2 (30%)")
                with col3:
                    status = "✓" if trade['tp3_closed'] else "○"
                    st.markdown(f"{status} TP3 (40%)")
                
                st.markdown("---")
                
                st.markdown("**Trade Management**")
                current_price = st.number_input("Current Price", value=float(trade['entry_price']), key=f"price_{trade['trade_id']}", help="Enter current market price for simulation")
                
                # Trade logic helper function
                def handle_tp_close(tp_level: str, target_price: float):
                    """Handle take profit closure"""
                    if st.button(f"Close {tp_level.upper()}", key=f"{tp_level}_{trade['trade_id']}", use_container_width=True):
                        try:
                            db.partial_close_trade(trade['trade_id'], current_price, tp_level)
                            get_cached_user_info.clear()
                            get_cached_stats.clear()
                            st.success(f"{tp_level.upper()} closed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to close {tp_level.upper()}: {e}")
                
                # LONG logic
                if trade['direction'] == 'LONG':
                    if not trade['tp1_closed'] and current_price >= trade['take_profit_1']:
                        st.success("🎯 TP1 Hit! Close 30% and move SL to breakeven?")
                        handle_tp_close('tp1', trade['take_profit_1'])
                    elif trade['tp1_closed'] and not trade['tp2_closed'] and current_price >= trade['take_profit_2']:
                        st.success("🎯 TP2 Hit! Close another 30%?")
                        handle_tp_close('tp2', trade['take_profit_2'])
                    elif trade['tp2_closed'] and not trade['tp3_closed'] and current_price >= trade['take_profit_3']:
                        st.success("🎯 TP3 Hit! Close remaining 40%?")
                        handle_tp_close('tp3', trade['take_profit_3'])
                    elif (trade['sl_breakeven'] and current_price <= trade['entry_price']) or (not trade['sl_breakeven'] and current_price <= trade['stop_loss']):
                        st.error("🛑 Stop Loss Triggered!")
                        if st.button("Close at SL", key=f"sl_{trade['trade_id']}", use_container_width=True):
                            db.full_close_trade(trade['trade_id'], current_price, "SL_HIT")
                            get_cached_user_info.clear()
                            get_cached_stats.clear()
                            st.rerun()
                
                # SHORT logic
                else:
                    if not trade['tp1_closed'] and current_price <= trade['take_profit_1']:
                        st.success("🎯 TP1 Hit! Close 30% and move SL to breakeven?")
                        handle_tp_close('tp1', trade['take_profit_1'])
                    elif trade['tp1_closed'] and not trade['tp2_closed'] and current_price <= trade['take_profit_2']:
                        st.success("🎯 TP2 Hit! Close another 30%?")
                        handle_tp_close('tp2', trade['take_profit_2'])
                    elif trade['tp2_closed'] and not trade['tp3_closed'] and current_price <= trade['take_profit_3']:
                        st.success("🎯 TP3 Hit! Close remaining 40%?")
                        handle_tp_close('tp3', trade['take_profit_3'])
                    elif (trade['sl_breakeven'] and current_price >= trade['entry_price']) or (not trade['sl_breakeven'] and current_price >= trade['stop_loss']):
                        st.error("🛑 Stop Loss Triggered!")
                        if st.button("Close at SL", key=f"sl_{trade['trade_id']}", use_container_width=True):
                            db.full_close_trade(trade['trade_id'], current_price, "SL_HIT")
                            get_cached_user_info.clear()
                            get_cached_stats.clear()
                            st.rerun()
                
                st.markdown("---")
                if st.button("Close Entire Position Manually", key=f"manual_{trade['trade_id']}", use_container_width=True):
                    try:
                        db.full_close_trade(trade['trade_id'], current_price, "MANUAL")
                        get_cached_user_info.clear()
                        get_cached_stats.clear()
                        st.success("Position closed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to close position: {e}")

# ========= TAB 4: Statistics =========
with tab_stats:
    st.markdown("### Performance Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", stats['total_trades'])
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    with col3:
        st.metric("Closed", stats['closed_trades'])
    with col4:
        st.metric("Avg P&L", f"{stats['avg_pnl']:.2f}%")
    
    st.markdown("---")
    
    closed_trades = db.get_closed_trades(st.session_state.user['user_id'])
    
    if closed_trades and pd:
        df = pd.DataFrame(closed_trades)
        df['cumulative_pnl'] = df['total_pnl_dollars'].cumsum()
        df['equity'] = user_info['initial_capital'] + df['cumulative_pnl']
        
        st.markdown("#### Equity Curve")
        try:
            st.line_chart(df.set_index('close_date')['equity'], use_container_width=True)
        except Exception:
            st.line_chart(df['equity'], use_container_width=True)
        
        st.markdown("#### Trade History")
        display_cols = ['ticker', 'direction', 'entry_price', 'close_price', 'close_reason', 'total_pnl_percent', 'total_pnl_dollars', 'close_date']
        
        try:
            styled_df = df[display_cols].style.format({
                'entry_price': '${:.2f}',
                'close_price': '${:.2f}',
                'total_pnl_percent': '{:.2f}%',
                'total_pnl_dollars': '${:.2f}'
            })
            st.dataframe(styled_df, use_container_width=True, height=400)
        except Exception:
            st.dataframe(df[display_cols], use_container_width=True, height=400)
    else:
        st.info("No closed trades yet. Close some active trades to see your performance history.")

st.markdown('</div>', unsafe_allow_html=True)

# ========= Footer =========
st.markdown("---")
col1, col2 = st.columns([4, 1])
with col1:
    st.caption("Arxora — Professional AI-Powered Trading Intelligence · 2025")
with col2:
    if st.button("Logout", use_container_width=True):
        # Clear all caches
        get_cached_user_info.clear()
        get_cached_stats.clear()
        # Clear session
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
