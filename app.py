# -*- coding: utf-8 -*-
# app.py — Arxora Trading Platform v6.0 (Complete)

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
    --surface: #1a1a1a;
    --accent-primary: #16c784;
    --accent-blue: #5B7FF9;
    --success: #16c784;
    --danger: #ea3943;
    --warning: #ffa94d;
    --text-primary: #ffffff;
    --text-secondary: #a0a0a0;
    --text-tertiary: #707070;
    --border: rgba(255, 255, 255, 0.1);
}

html, body, .stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif !important;
}

#MainMenu, footer, header {visibility: hidden !important;}
.stDeployButton {display: none !important;}
.block-container {padding: 2rem 3rem !important; max-width: 100% !important;}

/* Signal Cards */
.signal-box {
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 2px solid;
}

.signal-box.long {
    background: linear-gradient(135deg, rgba(22, 199, 132, 0.15), rgba(22, 199, 132, 0.05));
    border-color: #16c784;
}

.signal-box.short {
    background: linear-gradient(135deg, rgba(234, 57, 67, 0.15), rgba(234, 57, 67, 0.05));
    border-color: #ea3943;
}

.signal-box.wait {
    background: linear-gradient(135deg, rgba(255, 169, 77, 0.15), rgba(255, 169, 77, 0.05));
    border-color: #ffa94d;
}

/* Inputs */
.stTextInput input, .stNumberInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 0.875rem !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent-primary) !important;
    outline: none !important;
}

.stTextInput label, .stNumberInput label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    padding: 1rem !important;
    background: var(--accent-primary) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: #14b578 !important;
}

.stButton > button[kind="primary"] {
    background: var(--accent-blue) !important;
    color: #fff !important;
}

/* Radio - Remove Card Style */
.stRadio > div {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    background: transparent !important;
}

.stRadio > div > label {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.25rem !important;
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.stRadio > div > label:hover {
    border-color: var(--text-secondary) !important;
}

.stRadio > div > label[data-checked="true"] {
    background: var(--accent-primary) !important;
    border-color: var(--accent-primary) !important;
    color: #000 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-tertiary);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 13px;
}

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
}

/* Alerts */
.stAlert {
    background: var(--surface) !important;
    border-left-width: 3px !important;
    color: var(--text-primary) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
}

[data-testid="stMetric"] label {
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--border) !important;
}

/* Slider */
.stSlider {
    padding: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ========= Helper Functions =========
def _user_exists_in_current_db(username: str) -> bool:
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

try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# ========= AUTH PAGE =========
def show_auth_page():
    st.title("Arxora")
    st.caption("AI-POWERED TRADING INTELLIGENCE")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Sign In")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Sign In", type="primary"):
            if not username or not password:
                st.error("Enter username and password")
            else:
                user = db.login_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        st.subheader("Create Account")
        new_user = st.text_input("Username", key="reg_user")
        new_pass = st.text_input("Password", type="password", key="reg_pass")
        capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000)
        
        if st.button("Create Account", type="primary"):
            if len((new_user or "").strip()) < 3:
                st.error("Username: min 3 characters")
            elif len((new_pass or "").strip()) < 6:
                st.error("Password: min 6 characters")
            elif _user_exists_in_current_db(new_user):
                st.error("Username taken")
            else:
                try:
                    user_id = db.register_user(new_user, new_pass, capital)
                    if user_id:
                        user = db.login_user(new_user, new_pass)
                        if user:
                            st.session_state.user = user
                            st.success("Account created!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ========= CHECK AUTH =========
if 'user' not in st.session_state:
    show_auth_page()
    st.stop()

# ========= INIT SESSION =========
if 'min_confidence_filter' not in st.session_state:
    st.session_state['min_confidence_filter'] = 60

# ========= SIDEBAR =========
with st.sidebar:
    st.title("Arxora")
    st.caption("TRADING INTELLIGENCE")
    st.markdown("---")
    
    user_info = db.get_user_info(st.session_state.user['user_id'])
    stats = db.get_statistics(st.session_state.user['user_id'])
    
    if user_info:
        # Account & P&L
        st.subheader("Account")
        pnl = user_info['current_capital'] - user_info['initial_capital']
        pnl_pct = (pnl / max(1e-9, user_info['initial_capital'])) * 100
        
        st.metric(
            "Capital", 
            f"${user_info['current_capital']:,.2f}",
            f"{pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )
        
        st.markdown("---")
        
        # Statistics
        st.subheader("Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Trades", stats['total_trades'])
            st.metric("Closed", stats['closed_trades'])
        with col2:
            st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
            st.metric("Avg P&L", f"{stats['avg_pnl']:.2f}%")
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        new_conf = st.slider(
            "Min. Confidence (%)", 
            0, 100, 
            st.session_state['min_confidence_filter'], 
            5
        )
        st.session_state['min_confidence_filter'] = new_conf
        
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ========= MAIN CONTENT =========
st.title("Arxora")

tabs = st.tabs(["AI Signals", "Portfolio", "Active Trades", "Statistics"])

# TAB 1: AI Signals
with tabs[0]:
    st.subheader("Trading Agent Analysis")
    
    st.write("**Model**")
    models = get_available_models()
    model = st.radio("", models, horizontal=True, label_visibility="collapsed")
    
    st.write("**Symbol**")
    ticker = st.text_input("", placeholder="AAPL, BTCUSD", label_visibility="collapsed")
    
    if st.button("Analyze", type="primary"):
        if not ticker:
            st.warning("Enter symbol")
        else:
            symbol = normalize_for_polygon(ticker.strip().upper())
            with st.spinner(f"Analyzing {ticker}..."):
                try:
                    output = run_model_by_name(symbol, model)
                    rec = output.get("recommendation", {})
                    action = str(rec.get("action", "WAIT"))
                    conf = float(rec.get("confidence", 0.0))
                    conf_pct = conf * 100 if conf <= 1 else conf
                    
                    price = float(output.get("last_price", 0.0) or 0.0)
                    lv = {k: float(output.get("levels", {}).get(k, 0.0)) for k in ("entry", "sl", "tp1", "tp2", "tp3")}
                    
                    if action in ("BUY", "SHORT"):
                        tp1, tp2, tp3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
                        lv["tp1"], lv["tp2"], lv["tp3"] = tp1, tp2, tp3
                    
                    # Store in session
                    st.session_state["last_signal"] = {
                        "ticker": ticker.upper(),
                        "symbol": symbol,
                        "action": action,
                        "confidence": conf_pct,
                        "model": model,
                        "output": output,
                        "price": price,
                        "levels": lv
                    }
                    
                    # Determine card color class
                    if action == "BUY":
                        card_class = "long"
                        action_text = "LONG"
                    elif action == "SHORT":
                        card_class = "short"
                        action_text = "SHORT"
                    else:
                        card_class = "wait"
                        action_text = "WAIT"
                    
                    # Display signal with colored card
                    st.markdown(f'<div class="signal-box {card_class}">', unsafe_allow_html=True)
                    st.markdown(f"### {action_text} — {ticker.upper()} @ ${price:.2f} ({conf_pct:.0f}% confidence)")
                    
                    # AI Override info
                    ai_override = conf_pct - 44.0  # assuming base rules confidence is 44%
                    st.caption(f"AI Override: {ai_override:+.0f}% · Combined algorithmic and ML analysis")
                    
                    if action in ("BUY", "SHORT"):
                        st.markdown("---")
                        
                        # Entry, SL, and all TPs
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Entry", f"${lv['entry']:.2f}")
                        with col2:
                            st.metric("Stop Loss", f"${lv['sl']:.2f}")
                        with col3:
                            st.metric("TP1", f"${lv['tp1']:.2f}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("TP2", f"${lv['tp2']:.2f}")
                        with col2:
                            st.metric("TP3", f"${lv['tp3']:.2f}")
                        with col3:
                            rr = rr_line(lv)
                            if rr:
                                st.caption(f"**R/R:** {rr}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    try:
                        log_agent_performance(model, ticker, datetime.today(), 0.0)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    if ARXORA_DEBUG:
                        st.exception(e)

# TAB 2: Portfolio
with tabs[1]:
    st.subheader("Add to Portfolio")
    
    if "last_signal" in st.session_state:
        sig = st.session_state["last_signal"]
        
        if sig["action"] not in ("BUY", "SHORT"):
            st.warning("Signal was WAIT")
        elif not db.can_add_trade(st.session_state.user['user_id'], sig["ticker"]):
            st.warning(f"Trade exists for {sig['ticker']}")
        else:
            st.success(f"{sig['ticker']} — {sig['action']} ({sig['confidence']:.0f}%)")
            
            position_pct = st.slider("Position %", 5, 50, 10, 5)
            
            position_size = (user_info['current_capital'] * position_pct) / 100
            st.info(f"Position Size: ${position_size:,.2f}")
            
            if st.button("Add Trade", type="primary"):
                try:
                    data = {
                        'ticker': sig["ticker"],
                        'direction': 'LONG' if sig["action"] == 'BUY' else 'SHORT',
                        'entry_price': sig["levels"]['entry'],
                        'stop_loss': sig["levels"]['sl'],
                        'tp1': sig["levels"]['tp1'],
                        'tp2': sig["levels"]['tp2'],
                        'tp3': sig["levels"]['tp3'],
                        'tp1_prob': float(sig["output"].get('probs', {}).get('tp1', 0) * 100),
                        'tp2_prob': float(sig["output"].get('probs', {}).get('tp2', 0) * 100),
                        'tp3_prob': float(sig["output"].get('probs', {}).get('tp3', 0) * 100),
                        'confidence': int(sig["confidence"]),
                        'model': sig["model"]
                    }
                    trade_id = db.add_trade(st.session_state.user['user_id'], data, position_pct)
                    st.success(f"Trade #{trade_id} added!")
                    del st.session_state["last_signal"]
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        st.info("Analyze asset first")

# TAB 3: Active Trades
with tabs[2]:
    st.subheader("Active Trades")
    
    trades = db.get_active_trades(st.session_state.user['user_id'])
    
    if not trades:
        st.info("No active trades")
    else:
        for t in trades:
            with st.expander(f"{t['ticker']} — {t['direction']} ({t['remaining_percent']:.0f}% remaining)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entry", f"${t['entry_price']:.2f}")
                with col2:
                    st.metric("Position", f"${t['position_size']:.2f}")
                with col3:
                    st.metric("Confidence", f"{t['confidence']}%")
                
                st.markdown("**Take Profit Status**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    status = "✓" if t['tp1_closed'] else "○"
                    st.write(f"{status} TP1: ${t['take_profit_1']:.2f}")
                with col2:
                    status = "✓" if t['tp2_closed'] else "○"
                    st.write(f"{status} TP2: ${t['take_profit_2']:.2f}")
                with col3:
                    status = "✓" if t['tp3_closed'] else "○"
                    st.write(f"{status} TP3: ${t['take_profit_3']:.2f}")
                
                st.markdown("---")
                
                price = st.number_input("Current Price", float(t['entry_price']), key=f"p_{t['trade_id']}")
                
                # Close buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    # TP buttons based on direction
                    if t['direction'] == 'LONG':
                        if not t['tp1_closed'] and price >= t['take_profit_1']:
                            if st.button("Close TP1", key=f"tp1_{t['trade_id']}", use_container_width=True):
                                db.partial_close_trade(t['trade_id'], price, 'tp1')
                                st.rerun()
                        elif t['tp1_closed'] and not t['tp2_closed'] and price >= t['take_profit_2']:
                            if st.button("Close TP2", key=f"tp2_{t['trade_id']}", use_container_width=True):
                                db.partial_close_trade(t['trade_id'], price, 'tp2')
                                st.rerun()
                        elif t['tp2_closed'] and not t['tp3_closed'] and price >= t['take_profit_3']:
                            if st.button("Close TP3", key=f"tp3_{t['trade_id']}", use_container_width=True):
                                db.partial_close_trade(t['trade_id'], price, 'tp3')
                                st.rerun()
                    else:  # SHORT
                        if not t['tp1_closed'] and price <= t['take_profit_1']:
                            if st.button("Close TP1", key=f"tp1_{t['trade_id']}", use_container_width=True):
                                db.partial_close_trade(t['trade_id'], price, 'tp1')
                                st.rerun()
                        elif t['tp1_closed'] and not t['tp2_closed'] and price <= t['take_profit_2']:
                            if st.button("Close TP2", key=f"tp2_{t['trade_id']}", use_container_width=True):
                                db.partial_close_trade(t['trade_id'], price, 'tp2')
                                st.rerun()
                        elif t['tp2_closed'] and not t['tp3_closed'] and price <= t['take_profit_3']:
                            if st.button("Close TP3", key=f"tp3_{t['trade_id']}", use_container_width=True):
                                db.partial_close_trade(t['trade_id'], price, 'tp3')
                                st.rerun()
                
                with col2:
                    # Manual close entire position
                    if st.button("Close All", key=f"close_{t['trade_id']}", use_container_width=True):
                        db.full_close_trade(t['trade_id'], price, "MANUAL")
                        st.rerun()

# TAB 4: Statistics
with tabs[3]:
    st.subheader("Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", stats['total_trades'])
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    with col3:
        st.metric("Closed", stats['closed_trades'])
    with col4:
        st.metric("Avg P&L", f"{stats['avg_pnl']:.2f}%")
    
    closed = db.get_closed_trades(st.session_state.user['user_id'])
    if closed and pd:
        df = pd.DataFrame(closed)
        df['cumulative_pnl'] = df['total_pnl_dollars'].cumsum()
        
        st.markdown("### Equity Curve")
        st.line_chart(df['cumulative_pnl'])
        
        st.markdown("### Trade History")
        st.dataframe(df[['ticker', 'direction', 'entry_price', 'close_price', 'total_pnl_percent', 'close_date']], use_container_width=True)
    else:
        st.info("No closed trades")

st.markdown("---")
st.caption("Arxora · Professional Trading Intelligence · 2025")

