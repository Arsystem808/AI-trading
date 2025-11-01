# -*- coding: utf-8 -*-
# app.py — Arxora Trading Platform (Production v2.0 - SaxoTrader GO Inspired)

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

# ========= Model Loading (Non-blocking) =========
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

# ========= Database / Portfolio =========
try:
    from database import TradingDatabase
    db = TradingDatabase()
except Exception as e:
    st.error(f"⚠️ Database initialization failed: {e}")
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
    page_title="Arxora — AI Trading Platform",
    page_icon="assets/arxora_favicon_512.png",
    layout="centered",
    initial_sidebar_state="auto"
)

# ========= Premium SaxoTrader GO Inspired Theme =========
def inject_saxo_theme():
    """Inject premium SaxoTrader GO inspired CSS theme"""
    
    css = """
    <style>
    /* ==================== SAXOTRADER GO INSPIRED THEME ==================== */
    
    /* Root Variables */
    :root {
        --saxo-bg-primary: #0a0e13;
        --saxo-bg-secondary: #131920;
        --saxo-bg-tertiary: #1a2028;
        --saxo-surface: #1e2730;
        --saxo-surface-elevated: #242d38;
        
        --saxo-accent-blue: #0088ff;
        --saxo-accent-blue-dark: #0066cc;
        --saxo-accent-cyan: #00d4ff;
        
        --saxo-green: #00c896;
        --saxo-green-dark: #00a077;
        --saxo-red: #ff4757;
        --saxo-red-dark: #e63946;
        --saxo-yellow: #ffa94d;
        
        --saxo-text-primary: #ffffff;
        --saxo-text-secondary: #b8c5d0;
        --saxo-text-muted: #7b8794;
        --saxo-text-disabled: #4a5562;
        
        --saxo-border: rgba(255, 255, 255, 0.06);
        --saxo-border-light: rgba(255, 255, 255, 0.03);
        
        --saxo-shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
        --saxo-shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
        --saxo-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
        
        --saxo-radius-sm: 8px;
        --saxo-radius-md: 12px;
        --saxo-radius-lg: 16px;
    }
    
    /* Global Styles */
    * {
        font-feature-settings: "tnum" 1;
    }
    
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, var(--saxo-bg-primary) 0%, var(--saxo-bg-secondary) 100%) !important;
        color: var(--saxo-text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 15px;
        line-height: 1.5;
    }
    
    .block-container {
        max-width: 100% !important;
        padding: 1rem 1rem 2rem 1rem !important;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ==================== HEADER SECTION ==================== */
    
    .arxora-header {
        background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
        border-radius: var(--saxo-radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--saxo-border);
        box-shadow: var(--saxo-shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .arxora-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--saxo-accent-blue), var(--saxo-accent-cyan));
    }
    
    .arxora-logo {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff, #b8c5d0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        padding: 0;
    }
    
    .arxora-tagline {
        color: var(--saxo-text-muted);
        font-size: 13px;
        font-weight: 500;
        margin-top: 4px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* ==================== CARDS & PANELS ==================== */
    
    .saxo-card {
        background: var(--saxo-surface);
        border-radius: var(--saxo-radius-md);
        border: 1px solid var(--saxo-border);
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: var(--saxo-shadow-sm);
        transition: all 0.2s ease;
    }
    
    .saxo-card:hover {
        border-color: rgba(255, 255, 255, 0.1);
        box-shadow: var(--saxo-shadow-md);
    }
    
    .saxo-card-compact {
        background: var(--saxo-surface);
        border-radius: var(--saxo-radius-sm);
        border: 1px solid var(--saxo-border-light);
        padding: 0.875rem;
        margin-bottom: 0.75rem;
    }
    
    /* ==================== SIGNAL DISPLAY ==================== */
    
    .signal-container {
        background: var(--saxo-surface-elevated);
        border-radius: var(--saxo-radius-lg);
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid var(--saxo-border);
        box-shadow: var(--saxo-shadow-md);
    }
    
    .signal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        background: rgba(0, 136, 255, 0.08);
        border-radius: var(--saxo-radius-md);
        margin-bottom: 1.25rem;
        border: 1px solid rgba(0, 136, 255, 0.15);
    }
    
    .signal-header.long {
        background: linear-gradient(135deg, rgba(0, 200, 150, 0.12), rgba(0, 200, 150, 0.04));
        border-color: rgba(0, 200, 150, 0.2);
    }
    
    .signal-header.short {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.12), rgba(255, 71, 87, 0.04));
        border-color: rgba(255, 71, 87, 0.2);
    }
    
    .signal-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1rem;
        border-radius: var(--saxo-radius-sm);
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        min-width: 80px;
    }
    
    .signal-badge.long {
        background: var(--saxo-green);
        color: #ffffff;
    }
    
    .signal-badge.short {
        background: var(--saxo-red);
        color: #ffffff;
    }
    
    .signal-badge.wait {
        background: var(--saxo-text-disabled);
        color: var(--saxo-text-secondary);
    }
    
    .signal-direction {
        font-size: 20px;
        font-weight: 700;
        color: var(--saxo-text-primary);
    }
    
    .signal-confidence {
        font-size: 14px;
        color: var(--saxo-text-muted);
        font-weight: 600;
    }
    
    /* ==================== ASSET DISPLAY ==================== */
    
    .asset-title {
        font-size: clamp(18px, 4vw, 24px);
        font-weight: 700;
        text-align: center;
        color: var(--saxo-text-primary);
        margin: 0.5rem 0;
        letter-spacing: -0.3px;
    }
    
    .asset-price {
        font-size: clamp(32px, 6vw, 48px);
        font-weight: 800;
        text-align: center;
        color: var(--saxo-text-primary);
        margin: 0.75rem 0 1.5rem 0;
        font-feature-settings: "tnum" 1;
        letter-spacing: -1px;
    }
    
    /* ==================== PRICE LEVELS GRID ==================== */
    
    .levels-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.875rem;
        margin: 1.25rem 0;
    }
    
    .level-card {
        background: var(--saxo-surface);
        border-radius: var(--saxo-radius-md);
        padding: 1rem;
        border: 1px solid var(--saxo-border);
        transition: all 0.2s ease;
    }
    
    .level-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateY(-2px);
    }
    
    .level-card.entry {
        background: linear-gradient(135deg, rgba(0, 200, 150, 0.1), rgba(0, 200, 150, 0.05));
        border-color: rgba(0, 200, 150, 0.3);
    }
    
    .level-card.stoploss {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.1), rgba(255, 71, 87, 0.05));
        border-color: rgba(255, 71, 87, 0.3);
    }
    
    .level-card.takeprofit {
        background: linear-gradient(135deg, rgba(0, 136, 255, 0.1), rgba(0, 136, 255, 0.05));
        border-color: rgba(0, 136, 255, 0.2);
    }
    
    .level-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--saxo-text-muted);
        margin-bottom: 0.5rem;
    }
    
    .level-value {
        font-size: 22px;
        font-weight: 700;
        color: var(--saxo-text-primary);
        font-feature-settings: "tnum" 1;
    }
    
    .level-sub {
        font-size: 12px;
        color: var(--saxo-text-secondary);
        margin-top: 0.375rem;
    }
    
    /* ==================== CONFIDENCE METER ==================== */
    
    .confidence-meter {
        background: var(--saxo-bg-tertiary);
        border-radius: var(--saxo-radius-md);
        padding: 1rem;
        margin: 1.25rem 0;
        border: 1px solid var(--saxo-border);
        font-family: "SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace;
    }
    
    .confidence-bar-container {
        position: relative;
        height: 32px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: var(--saxo-radius-sm);
        margin: 0.75rem 0;
        overflow: hidden;
    }
    
    .confidence-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--saxo-green), var(--saxo-accent-cyan));
        transition: width 0.6s ease;
        border-radius: var(--saxo-radius-sm);
    }
    
    .confidence-text {
        font-size: 13px;
        color: var(--saxo-text-secondary);
        line-height: 1.6;
    }
    
    .confidence-value {
        font-size: 18px;
        font-weight: 700;
        color: var(--saxo-text-primary);
    }
    
    /* ==================== BUTTONS ==================== */
    
    .stButton > button {
        background: linear-gradient(135deg, var(--saxo-accent-blue), var(--saxo-accent-blue-dark)) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--saxo-radius-md) !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--saxo-shadow-sm) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--saxo-accent-blue-dark), #004d99) !important;
        box-shadow: var(--saxo-shadow-md) !important;
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--saxo-green), var(--saxo-green-dark)) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--saxo-green-dark), #008866) !important;
    }
    
    /* ==================== INPUTS ==================== */
    
    .stTextInput input, .stNumberInput input {
        background: var(--saxo-surface) !important;
        border: 1px solid var(--saxo-border) !important;
        border-radius: var(--saxo-radius-sm) !important;
        color: var(--saxo-text-primary) !important;
        font-size: 15px !important;
        padding: 0.625rem 0.875rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--saxo-accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(0, 136, 255, 0.1) !important;
    }
    
    .stTextInput input::placeholder {
        color: var(--saxo-text-disabled) !important;
    }
    
    /* ==================== TABS ==================== */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--saxo-surface);
        padding: 0.5rem;
        border-radius: var(--saxo-radius-md);
        border: 1px solid var(--saxo-border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: var(--saxo-text-muted);
        font-weight: 600;
        padding: 0.625rem 1rem;
        border-radius: var(--saxo-radius-sm);
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: var(--saxo-text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--saxo-accent-blue) !important;
        color: white !important;
    }
    
    /* ==================== METRICS ==================== */
    
    .stMetric {
        background: var(--saxo-surface);
        border-radius: var(--saxo-radius-md);
        padding: 1rem;
        border: 1px solid var(--saxo-border);
    }
    
    .stMetric label {
        font-size: 12px;
        font-weight: 600;
        color: var(--saxo-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        color: var(--saxo-text-primary);
    }
    
    /* ==================== DATAFRAMES ==================== */
    
    .stDataFrame {
        border-radius: var(--saxo-radius-md);
        overflow: hidden;
    }
    
    .stDataFrame table {
        background: var(--saxo-surface) !important;
        color: var(--saxo-text-primary) !important;
    }
    
    .stDataFrame th {
        background: var(--saxo-bg-tertiary) !important;
        color: var(--saxo-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        padding: 0.875rem !important;
        border-bottom: 2px solid var(--saxo-border) !important;
    }
    
    .stDataFrame td {
        padding: 0.875rem !important;
        border-bottom: 1px solid var(--saxo-border-light) !important;
    }
    
    /* ==================== EXPANDER ==================== */
    
    .streamlit-expanderHeader {
        background: var(--saxo-surface) !important;
        border: 1px solid var(--saxo-border) !important;
        border-radius: var(--saxo-radius-md) !important;
        color: var(--saxo-text-primary) !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--saxo-surface-elevated) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--saxo-surface) !important;
        border: 1px solid var(--saxo-border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--saxo-radius-md) var(--saxo-radius-md) !important;
        padding: 1rem !important;
    }
    
    /* ==================== SIDEBAR ==================== */
    
    [data-testid="stSidebar"] {
        background: var(--saxo-bg-secondary) !important;
        border-right: 1px solid var(--saxo-border);
    }
    
    [data-testid="stSidebar"] .stMetric {
        background: var(--saxo-surface-elevated);
    }
    
    /* ==================== INFO/WARNING/ERROR BOXES ==================== */
    
    .stInfo, .stWarning, .stError, .stSuccess {
        border-radius: var(--saxo-radius-md) !important;
        border-left-width: 4px !important;
        padding: 1rem !important;
    }
    
    .stInfo {
        background: rgba(0, 136, 255, 0.1) !important;
        border-left-color: var(--saxo-accent-blue) !important;
    }
    
    .stWarning {
        background: rgba(255, 169, 77, 0.1) !important;
        border-left-color: var(--saxo-yellow) !important;
    }
    
    .stError {
        background: rgba(255, 71, 87, 0.1) !important;
        border-left-color: var(--saxo-red) !important;
    }
    
    .stSuccess {
        background: rgba(0, 200, 150, 0.1) !important;
        border-left-color: var(--saxo-green) !important;
    }
    
    /* ==================== RADIO BUTTONS ==================== */
    
    .stRadio > div {
        background: var(--saxo-surface);
        border-radius: var(--saxo-radius-md);
        padding: 0.75rem;
        border: 1px solid var(--saxo-border);
    }
    
    .stRadio label {
        color: var(--saxo-text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* ==================== SLIDER ==================== */
    
    .stSlider {
        padding: 0.5rem 0;
    }
    
    /* ==================== MOBILE OPTIMIZATIONS ==================== */
    
    @media (max-width: 768px) {
        .block-container {
            padding: 0.75rem 0.75rem 1.5rem 0.75rem !important;
        }
        
        .arxora-header {
            padding: 1rem;
        }
        
        .arxora-logo {
            font-size: 24px;
        }
        
        .asset-price {
            font-size: 36px;
        }
        
        .levels-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.625rem;
        }
        
        .level-card {
            padding: 0.875rem;
        }
        
        .level-value {
            font-size: 18px;
        }
        
        .saxo-card {
            padding: 1rem;
        }
    }
    
    /* ==================== ANIMATIONS ==================== */
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .signal-container {
        animation: fadeIn 0.3s ease;
    }
    
    /* ==================== UTILITY CLASSES ==================== */
    
    .text-muted {
        color: var(--saxo-text-muted);
        font-size: 13px;
    }
    
    .text-secondary {
        color: var(--saxo-text-secondary);
    }
    
    .text-center {
        text-align: center;
    }
    
    .font-mono {
        font-family: "SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace;
    }
    
    .divider {
        height: 1px;
        background: var(--saxo-border);
        margin: 1.5rem 0;
    }
    
    /* ==================== SCROLLBAR ==================== */
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--saxo-bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--saxo-text-disabled);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--saxo-text-muted);
    }
    
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

# Inject theme
inject_saxo_theme()

# ========= Header Component =========
def render_premium_header():
    """Render premium SaxoTrader-inspired header"""
    st.markdown('''
        <div class="arxora-header">
            <div class="arxora-logo">Arxora</div>
            <div class="arxora-tagline">AI-Powered Trading Intelligence</div>
        </div>
    ''', unsafe_allow_html=True)

# ========= Helper Functions =========
def _user_exists_in_current_db(username: str) -> bool:
    """Check if user exists in current database"""
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

# ========= Authentication Page =========
def show_auth_page():
    """Display authentication page with login and registration"""
    render_premium_header()
    
    st.markdown('<div class="saxo-card">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "✨ Register"])
    
    with tab1:
        st.markdown("### Sign In to Your Account")
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Sign In", type="primary", use_container_width=True):
                user = db.login_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
                    if username and not _user_exists_in_current_db(username):
                        st.info("💡 User not found. Please register first.")
    
    with tab2:
        st.markdown("### Create New Account")
        new_username = st.text_input("Username", key="reg_username", placeholder="Choose a username (min 3 characters)")
        new_password = st.text_input("Password", type="password", key="reg_password", placeholder="Choose a password (min 6 characters)")
        initial_capital = st.number_input("Initial Capital (Virtual)", min_value=1000, value=10000, step=1000)
        
        if st.button("✨ Create Account", type="primary", use_container_width=True):
            if len((new_username or "").strip()) < 3:
                st.error("❌ Username must be at least 3 characters")
            elif len((new_password or "").strip()) < 6:
                st.error("❌ Password must be at least 6 characters")
            else:
                user_id = db.register_user(new_username, new_password, initial_capital)
                if user_id:
                    user = db.login_user(new_username, new_password)
                    if user:
                        st.session_state.user = user
                        st.success("✅ Account created! Welcome aboard!")
                        st.balloons()
                        st.rerun()
                else:
                    st.error("❌ Username already taken")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========= Require Authentication =========
if 'user' not in st.session_state:
    show_auth_page()
    st.stop()

# ========= Sidebar =========
user_info = db.get_user_info(st.session_state.user['user_id'])
stats = db.get_statistics(st.session_state.user['user_id'])

st.sidebar.markdown(f"### 👤 {user_info['username']}")
st.sidebar.metric("Current Capital", f"${user_info['current_capital']:,.2f}")
st.sidebar.metric("Initial Capital", f"${user_info['initial_capital']:,.2f}")

pnl_change = user_info['current_capital'] - user_info['initial_capital']
pnl_percent = (pnl_change / max(1e-9, user_info['initial_capital'])) * 100
st.sidebar.metric("Total P&L", f"${pnl_change:,.2f}", f"{pnl_percent:.2f}%")

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Settings")
min_confidence_filter = st.sidebar.slider("Min. Confidence (%)", 0, 100, 60)

st.sidebar.divider()
if st.sidebar.button("🚪 Logout", use_container_width=True):
    del st.session_state.user
    st.rerun()

# ========= Performance Tracking =========
try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# ========= Domain Helpers =========
def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "0.00"

def sanitize_targets(action: str, entry: float, tp1: float, tp2: float, tp3: float):
    """Sanitize and order take profit targets"""
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
    """Determine entry mode (Market/Limit/Stop)"""
    if action not in ("BUY", "SHORT"):
        return "WAIT", "Entry"
    if abs(entry - last_price) <= eps * max(1.0, abs(last_price)):
        return "Market", "Entry (Market)"
    if action == "BUY":
        return ("Buy Stop", "Entry (Buy Stop)") if entry > last_price else ("Buy Limit", "Entry (Buy Limit)")
    else:
        return ("Sell Stop", "Entry (Sell Stop)") if entry < last_price else ("Sell Limit", "Entry (Sell Limit)")

def normalize_for_polygon(symbol: str) -> str:
    """Normalize symbol for Polygon API"""
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
    """Calculate risk/reward ratios"""
    risk = abs(levels["entry"] - levels["sl"])
    if risk <= 1e-9:
        return ""
    rr1 = abs(levels["tp1"] - levels["entry"]) / risk
    rr2 = abs(levels["tp2"] - levels["entry"]) / risk
    rr3 = abs(levels["tp3"] - levels["entry"]) / risk
    return f"R:R → 1:{rr1:.1f} (TP1) · 1:{rr2:.1f} (TP2) · 1:{rr3:.1f} (TP3)"

@st.cache_data(show_spinner=False, ttl=86400)
def resolve_asset_title_polygon(raw_symbol: str, normalized: str) -> str:
    """Resolve asset full name from Polygon API"""
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

# ========= Strategy Module Loading =========
try:
    import services.data  # noqa
except Exception:
    try:
        import core.data as _core_data
        sys.modules['services.data'] = _core_data
    except Exception:
        pass

def _load_strategy_module():
    """Load strategy module dynamically"""
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
    """Get list of available trading models"""
    mod, _ = _load_strategy_module()
    if not mod:
        return ["Octopus"]
    reg = getattr(mod, "STRATEGY_REGISTRY", {}) or {}
    keys = list(reg.keys())
    return (["Octopus"] if "Octopus" in keys else []) + [k for k in sorted(keys) if k != "Octopus"]

def run_model_by_name(ticker_norm: str, model_name: str) -> Dict[str, Any]:
    """Execute trading model analysis"""
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

# ========= Confidence Breakdown Component =========
def render_confidence_meter(conf_pct: float, ai_override: float = 0.0):
    """Render SaxoTrader-style confidence meter"""
    conf_pct = max(0.0, min(100.0, conf_pct))
    
    html = f'''
    <div class="confidence-meter">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span class="text-muted">Confidence Level</span>
            <span class="confidence-value">{conf_pct:.0f}%</span>
        </div>
        <div class="confidence-bar-container">
            <div class="confidence-bar-fill" style="width: {conf_pct}%"></div>
        </div>
        <div class="confidence-text">
            AI Override: {ai_override:+.0f}% · Combined algorithmic and machine learning analysis
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

# ========= Signal Display Component =========
def render_signal_card(ticker: str, action: str, conf_pct: float, last_price: float, 
                       levels: Dict[str, float], model: str, output: Dict[str, Any]):
    """Render premium signal card with all trade details"""
    
    # Resolve asset title
    symbol_for_engine = normalize_for_polygon(ticker)
    asset_title = resolve_asset_title_polygon(ticker, symbol_for_engine)
    
    # Determine signal direction CSS class
    direction_class = "long" if action == "BUY" else ("short" if action == "SHORT" else "wait")
    badge_class = direction_class
    
    # Entry mode
    mode_text, entry_title = entry_mode_labels(action, levels.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
    
    # Build signal header text
    if action == "BUY":
        header_text = f"LONG · {mode_text}"
    elif action == "SHORT":
        header_text = f"SHORT · {mode_text}"
    else:
        header_text = "WAIT"
    
    # UTC timestamps
    now_utc = datetime.now(timezone.utc)
    eod_utc = now_utc.replace(hour=23, minute=59, second=59, microsecond=0)
    
    # Start signal container
    st.markdown('<div class="signal-container">', unsafe_allow_html=True)
    
    # Asset title and price
    st.markdown(f'<div class="asset-title">{asset_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="asset-price">${last_price:.2f}</div>', unsafe_allow_html=True)
    
    # Signal header
    st.markdown(f'''
        <div class="signal-header {direction_class}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="signal-badge {badge_class}">
                    {"LONG" if action == "BUY" else ("SHORT" if action == "SHORT" else "WAIT")}
                </div>
                <div class="signal-direction">{header_text}</div>
            </div>
            <div class="signal-confidence">{int(round(conf_pct))}%</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Timestamp
    st.markdown(
        f'<div class="text-muted text-center" style="margin-bottom: 1.25rem;">'
        f'Analysis: {now_utc.strftime("%Y-%m-%d %H:%M UTC")} · Valid until: {eod_utc.strftime("%H:%M UTC")} · Model: {model}'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Confidence meter
    ai_delta = float(st.session_state.get("last_overall_conf_pct", conf_pct)) - float(st.session_state.get("last_rules_pct", 44.0))
    render_confidence_meter(conf_pct, ai_delta)
    
    # Price levels
    if action in ("BUY", "SHORT"):
        probs = output.get('probs', {})
        
        # Levels grid
        st.markdown('<div class="levels-grid">', unsafe_allow_html=True)
        
        # Entry
        st.markdown(f'''
            <div class="level-card entry">
                <div class="level-label">{entry_title}</div>
                <div class="level-value">${levels["entry"]:.2f}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Stop Loss
        st.markdown(f'''
            <div class="level-card stoploss">
                <div class="level-label">Stop Loss</div>
                <div class="level-value">${levels["sl"]:.2f}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # TP1
        st.markdown(f'''
            <div class="level-card takeprofit">
                <div class="level-label">Take Profit 1</div>
                <div class="level-value">${levels["tp1"]:.2f}</div>
                <div class="level-sub">Probability: {int(round(probs.get('tp1', 0)*100))}%</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # TP2
        st.markdown(f'''
            <div class="level-card takeprofit">
                <div class="level-label">Take Profit 2</div>
                <div class="level-value">${levels["tp2"]:.2f}</div>
                <div class="level-sub">Probability: {int(round(probs.get('tp2', 0)*100))}%</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # TP3
        st.markdown(f'''
            <div class="level-card takeprofit">
                <div class="level-label">Take Profit 3</div>
                <div class="level-value">${levels["tp3"]:.2f}</div>
                <div class="level-sub">Probability: {int(round(probs.get('tp3', 0)*100))}%</div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close levels-grid
        
        # Risk/Reward
        rr = rr_line(levels)
        if rr:
            st.markdown(
                f'<div class="text-muted text-center" style="margin-top: 1rem; font-weight: 600;">{rr}</div>',
                unsafe_allow_html=True
            )
    
    # Market context
    ctx_phrases = {
        "BUY": "Support zone detected. Optimal for long positions with defined risk management. Monitor for breakout confirmation.",
        "SHORT": "Resistance zone identified. Favorable for short positions. Watch for breakdown continuation or rejection.",
        "WAIT": "Neutral market conditions. Await clear directional signal before entering position."
    }
    
    st.markdown(
        f'<div class="saxo-card-compact" style="margin-top: 1.25rem;">{ctx_phrases.get(action, ctx_phrases["WAIT"])}</div>',
        unsafe_allow_html=True
    )
    
    # Disclaimer
    st.caption(
        "⚠️ AI analysis is for informational purposes only and does not constitute investment advice. "
        "Markets are dynamic; past performance does not guarantee future results."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close signal-container

# ========= Main App =========
render_premium_header()

# Tabs
tab_signals, tab_portfolio, tab_active, tab_stats = st.tabs([
    "🎯 AI Signals",
    "💼 Portfolio",
    "📋 Active Trades",
    "📊 Statistics"
])

# ========= TAB 1: AI Signals =========
with tab_signals:
    st.markdown('<div class="saxo-card">', unsafe_allow_html=True)
    
    st.markdown("### 🤖 Trading Agent Analysis")
    
    # Model selection
    models = get_available_models()
    model = st.radio(
        "Select AI Model",
        options=models,
        index=0,
        horizontal=True,
        key="agent_radio"
    )
    
    # Ticker input
    ticker_input = st.text_input(
        "Asset Symbol",
        placeholder="Examples: AAPL · SPY · BTCUSD · C:EURUSD",
        help="Enter stock ticker, crypto pair, or forex symbol"
    )
    ticker = ticker_input.strip().upper()
    symbol_for_engine = normalize_for_polygon(ticker)
    
    # Analyze button
    if st.button("🔍 Analyze Asset", type="primary", use_container_width=True):
        if not ticker:
            st.warning("⚠️ Please enter an asset symbol")
        else:
            with st.spinner(f"Analyzing {ticker} with {model}..."):
                try:
                    # Run analysis
                    output = run_model_by_name(symbol_for_engine, model)
                    
                    # Extract recommendation
                    rec = output.get("recommendation")
                    if not rec and ("action" in output or "confidence" in output):
                        rec = {
                            "action": output.get("action", "WAIT"),
                            "confidence": float(output.get("confidence", 0.0))
                        }
                    if not rec:
                        rec = {"action": "WAIT", "confidence": 0.0}
                    
                    action = str(rec.get("action", "WAIT"))
                    conf_val = float(rec.get("confidence", 0.0))
                    conf_pct_val = conf_val * 100.0 if conf_val <= 1.0 else conf_val
                    
                    last_price = float(output.get("last_price", 0.0) or 0.0)
                    
                    # Extract levels
                    lv = {
                        k: float(output.get("levels", {}).get(k, 0.0))
                        for k in ("entry", "sl", "tp1", "tp2", "tp3")
                    }
                    
                    # Sanitize targets
                    if action in ("BUY", "SHORT"):
                        tp1, tp2, tp3 = sanitize_targets(
                            action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"]
                        )
                        lv["tp1"], lv["tp2"], lv["tp3"] = float(tp1), float(tp2), float(tp3)
                    
                    # Store in session
                    st.session_state["last_overall_conf_pct"] = conf_pct_val
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
                    
                    # Render signal card
                    render_signal_card(
                        ticker=ticker,
                        action=action,
                        conf_pct=conf_pct_val,
                        last_price=last_price,
                        levels=lv,
                        model=model,
                        output=output
                    )
                    
                    # Log performance
                    try:
                        log_agent_performance(model, ticker, datetime.today(), 0.0)
                    except Exception:
                        pass
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    if ARXORA_DEBUG:
                        st.exception(e)
    else:
        st.info(
            "💡 Enter an asset symbol and click **Analyze** to generate AI-powered trading signals. "
            "Our advanced algorithms will provide entry points, stop losses, and take profit targets."
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========= TAB 2: Portfolio =========
with tab_portfolio:
    st.markdown('<div class="saxo-card">', unsafe_allow_html=True)
    st.markdown("### 💼 Add Signal to Portfolio")
    
    if "last_signal" in st.session_state:
        sig = st.session_state["last_signal"]
        ticker = sig["ticker"]
        action = sig["action"]
        conf = sig["confidence"]
        lv = sig["levels"]
        
        if action not in ("BUY", "SHORT"):
            st.warning("⚠️ Last signal was WAIT. Cannot add to portfolio.")
        elif not db.can_add_trade(st.session_state.user['user_id'], ticker):
            st.warning(f"⚠️ Active trade already exists for {ticker}. Close it before adding a new one.")
        else:
            st.success(f"✅ Signal Ready: **{ticker}** — **{action}** ({conf:.0f}% confidence)")
            
            # Trade parameters
            st.markdown("#### 📋 Trade Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry", f"${lv['entry']:.2f}")
            with col2:
                st.metric("Stop Loss", f"${lv['sl']:.2f}")
            with col3:
                risk_pct = abs(lv['entry'] - lv['sl']) / max(1e-9, abs(lv['entry'])) * 100
                st.metric("Risk %", f"{risk_pct:.2f}%")
            
            st.markdown("#### 💰 Position Sizing")
            position_percent = st.slider(
                "Portfolio Allocation (%)",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Percentage of your capital to allocate to this trade"
            )
            
            position_size = (user_info['current_capital'] * position_percent) / 100
            st.info(f"📊 Position Size: **${position_size:,.2f}** ({position_percent}% of capital)")
            
            # P&L projections
            st.markdown("#### 📈 Profit/Loss Projections")
            potential_profit = position_size * abs(lv['tp1'] - lv['entry']) / max(1e-9, abs(lv['entry']))
            potential_loss = position_size * abs(lv['entry'] - lv['sl']) / max(1e-9, abs(lv['entry']))
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"💚 Potential Profit (TP1): **${potential_profit:.2f}**")
            with col2:
                st.error(f"💔 Potential Loss (SL): **${potential_loss:.2f}**")
            
            # Confidence warning
            if conf < min_confidence_filter:
                st.warning(
                    f"⚠️ **Confidence Alert**: Signal confidence ({conf:.0f}%) is below your filter ({min_confidence_filter}%). "
                    "Consider waiting for a stronger signal."
                )
            
            # Add to portfolio button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ ADD TO PORTFOLIO", type="primary", use_container_width=True):
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
                    trade_id = db.add_trade(
                        st.session_state.user['user_id'],
                        signal_data,
                        position_percent
                    )
                    st.success(f"🎉 Trade successfully added! Trade ID: #{trade_id}")
                    st.balloons()
                    del st.session_state["last_signal"]
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    else:
        st.info(
            "📊 **No signal available.** Please analyze an asset in the **AI Signals** tab first, "
            "then return here to add it to your portfolio."
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========= TAB 3: Active Trades =========
with tab_active:
    st.markdown('<div class="saxo-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Active Trades")
    
    active_trades = db.get_active_trades(st.session_state.user['user_id'])
    
    if not active_trades:
        st.info("📭 No active trades. Add a signal from the **Portfolio** tab to get started!")
    else:
        for trade in active_trades:
            sl_status = "Breakeven" if trade['sl_breakeven'] else f"${trade['stop_loss']:.2f}"
            
            # Trade header for expander
            direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
            header = f"{direction_icon} {trade['ticker']} — {trade['direction']} | {trade['remaining_percent']:.0f}% remaining | SL: {sl_status}"
            
            with st.expander(header, expanded=False):
                # Trade details
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entry", f"${trade['entry_price']:.2f}")
                with col2:
                    st.metric("Position", f"${trade['position_size']:.2f}")
                with col3:
                    st.metric("Model", trade['model_used'])
                with col4:
                    st.metric("Confidence", f"{trade['confidence']}%")
                
                # Progress indicators
                st.markdown("**📊 Take Profit Progress:**")
                progress_col1, progress_col2, progress_col3 = st.columns(3)
                with progress_col1:
                    status = "✅" if trade['tp1_closed'] else "⏳"
                    st.markdown(f"{status} **TP1** (30%)")
                with progress_col2:
                    status = "✅" if trade['tp2_closed'] else "⏳"
                    st.markdown(f"{status} **TP2** (30%)")
                with progress_col3:
                    status = "✅" if trade['tp3_closed'] else "⏳"
                    st.markdown(f"{status} **TP3** (40%)")
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                # Price input for simulation
                st.markdown("**💹 Trade Management**")
                current_price = st.number_input(
                    "Current Price (for simulation)",
                    value=float(trade['entry_price']),
                    key=f"price_{trade['trade_id']}",
                    help="Enter current market price to simulate trade closures"
                )
                
                # Trade logic for LONG
                if trade['direction'] == 'LONG':
                    if not trade['tp1_closed'] and current_price >= trade['take_profit_1']:
                        st.success("🎯 **TP1 Hit!** Close 30% and move SL to breakeven?")
                        if st.button("Close TP1", key=f"tp1_{trade['trade_id']}", use_container_width=True):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp1')
                            st.rerun()
                    
                    elif trade['tp1_closed'] and not trade['tp2_closed'] and current_price >= trade['take_profit_2']:
                        st.success("🎯 **TP2 Hit!** Close another 30%?")
                        if st.button("Close TP2", key=f"tp2_{trade['trade_id']}", use_container_width=True):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp2')
                            st.rerun()
                    
                    elif trade['tp2_closed'] and not trade['tp3_closed'] and current_price >= trade['take_profit_3']:
                        st.success("🎯 **TP3 Hit!** Close remaining 40%?")
                        if st.button("Close TP3", key=f"tp3_{trade['trade_id']}", use_container_width=True):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp3')
                            st.rerun()
                    
                    elif (trade['sl_breakeven'] and current_price <= trade['entry_price']) or \
                         (not trade['sl_breakeven'] and current_price <= trade['stop_loss']):
                        st.error("🛑 **Stop Loss Triggered!**")
                        if st.button("Close at SL", key=f"sl_{trade['trade_id']}", use_container_width=True):
                            db.full_close_trade(trade['trade_id'], current_price, "SL_HIT")
                            st.rerun()
                
                # Trade logic for SHORT
                elif trade['direction'] == 'SHORT':
                    if not trade['tp1_closed'] and current_price <= trade['take_profit_1']:
                        st.success("🎯 **TP1 Hit!** Close 30% and move SL to breakeven?")
                        if st.button("Close TP1", key=f"tp1_{trade['trade_id']}", use_container_width=True):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp1')
                            st.rerun()
                    
                    elif trade['tp1_closed'] and not trade['tp2_closed'] and current_price <= trade['take_profit_2']:
                        st.success("🎯 **TP2 Hit!** Close another 30%?")
                        if st.button("Close TP2", key=f"tp2_{trade['trade_id']}", use_container_width=True):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp2')
                            st.rerun()
                    
                    elif trade['tp2_closed'] and not trade['tp3_closed'] and current_price <= trade['take_profit_3']:
                        st.success("🎯 **TP3 Hit!** Close remaining 40%?")
                        if st.button("Close TP3", key=f"tp3_{trade['trade_id']}", use_container_width=True):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp3')
                            st.rerun()
                    
                    elif (trade['sl_breakeven'] and current_price >= trade['entry_price']) or \
                         (not trade['sl_breakeven'] and current_price >= trade['stop_loss']):
                        st.error("🛑 **Stop Loss Triggered!**")
                        if st.button("Close at SL", key=f"sl_{trade['trade_id']}", use_container_width=True):
                            db.full_close_trade(trade['trade_id'], current_price, "SL_HIT")
                            st.rerun()
                
                # Manual close option
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                if st.button("🔴 Close Entire Position Manually", key=f"manual_{trade['trade_id']}", use_container_width=True):
                    db.full_close_trade(trade['trade_id'], current_price, "MANUAL")
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========= TAB 4: Statistics =========
with tab_stats:
    st.markdown('<div class="saxo-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Performance Statistics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", stats['total_trades'])
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    with col3:
        st.metric("Closed Trades", stats['closed_trades'])
    with col4:
        st.metric("Avg P&L", f"{stats['avg_pnl']:.2f}%")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Trade history
    closed_trades = db.get_closed_trades(st.session_state.user['user_id'])
    
    if closed_trades and pd:
        df = pd.DataFrame(closed_trades)
        
        # Calculate cumulative metrics
        df['cumulative_pnl'] = df['total_pnl_dollars'].cumsum()
        df['equity'] = user_info['initial_capital'] + df['cumulative_pnl']
        
        # Equity curve
        st.markdown("### 📈 Equity Curve")
        try:
            st.line_chart(df.set_index('close_date')['equity'])
        except Exception:
            st.line_chart(df['equity'])
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Trade history table
        st.markdown("### 📜 Trade History")
        display_cols = [
            'ticker', 'direction', 'entry_price', 'close_price',
            'close_reason', 'total_pnl_percent', 'total_pnl_dollars', 'close_date'
        ]
        
        try:
            styled_df = df[display_cols].style.format({
                'entry_price': '${:.2f}',
                'close_price': '${:.2f}',
                'total_pnl_percent': '{:.2f}%',
                'total_pnl_dollars': '${:.2f}'
            })
            st.dataframe(styled_df, use_container_width=True)
        except Exception:
            st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("📭 No closed trades yet. Close some active trades to see your performance history here.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========= Footer =========
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('''
<div class="saxo-card-compact">
    <h4 style="margin: 0 0 0.5rem 0; font-weight: 600;">About Arxora</h4>
    <p class="text-muted" style="margin: 0; line-height: 1.6;">
        Arxora is a professional AI-powered trading platform that combines multiple algorithmic strategies
        with machine learning to deliver precise, actionable trading signals. Our ensemble approach
        analyzes market conditions from multiple perspectives, providing you with confidence-calibrated
        recommendations and comprehensive risk management tools. Features include trend-following and
        counter-trend strategies, the Octopus orchestrator for unified analysis, and AI Override
        technology for dynamic market adaptation.
    </p>
</div>
''', unsafe_allow_html=True)

st.markdown(
    '<div class="text-muted text-center" style="margin-top: 1.5rem; font-size: 12px;">'
    'Arxora v2.0 Production · Powered by Advanced AI · © 2025'
    '</div>',
    unsafe_allow_html=True
)
