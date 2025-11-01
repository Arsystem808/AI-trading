# -*- coding: utf-8 -*-
# app.py — Arxora Trading Platform v8.0 (Final Polish)

import os
import re
import sys
import time
import importlib
import traceback
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

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
.block-container {padding: 1.5rem 2.5rem !important; max-width: 100% !important;}

/* Inputs */
.stTextInput input, .stNumberInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 0.75rem !important;
    font-size: 14px !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent-primary) !important;
    outline: none !important;
}

.stTextInput label, .stNumberInput label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
    margin-bottom: 0.5rem !important;
}

/* Buttons - SMALLER SIZE */
.stButton > button {
    padding: 0.65rem 1.5rem !important;
    background: var(--accent-primary) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
}

.stButton > button:hover {
    background: #14b578 !important;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: var(--accent-blue) !important;
    color: #fff !important;
}

.stButton > button[kind="secondary"] {
    background: var(--danger) !important;
    color: #fff !important;
}

/* Radio */
.stRadio > div {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.stRadio > div > label {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.1rem !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
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
    font-size: 12px;
    padding: 0.75rem 0 !important;
}

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
}

[data-testid="stMetric"] label {
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    font-size: 10px !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
}

/* Headings */
h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}

h2 {
    font-size: 20px !important;
    font-weight: 700 !important;
    margin-bottom: 0.75rem !important;
    margin-top: 1rem !important;
}

h3 {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Signal Card Container */
.signal-container {
    margin: 1.5rem 0;
}

</style>
""", unsafe_allow_html=True)

# ========= Helper Functions =========
def clear_all_caches():
    """Clear all Streamlit caches"""
    st.cache_data.clear()

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

def get_tp_status(trade: Dict, price: float) -> Tuple[Optional[str], bool]:
    """Get which TP can be closed - DRY principle"""
    is_long = trade['direction'] == 'LONG'
    
    if not trade['tp1_closed']:
        can_close = (price >= trade['take_profit_1']) if is_long else (price <= trade['take_profit_1'])
        return ('tp1', can_close)
    elif not trade['tp2_closed']:
        can_close = (price >= trade['take_profit_2']) if is_long else (price <= trade['take_profit_2'])
        return ('tp2', can_close)
    elif not trade['tp3_closed']:
        can_close = (price >= trade['take_profit_3']) if is_long else (price <= trade['take_profit_3'])
        return ('tp3', can_close)
    
    return (None, False)

def check_sl_hit(trade: Dict, price: float) -> bool:
    """Check if stop loss is hit"""
    is_long = trade['direction'] == 'LONG'
    
    if trade['sl_breakeven']:
        return (price <= trade['entry_price']) if is_long else (price >= trade['entry_price'])
    else:
        return (price <= trade['stop_loss']) if is_long else (price >= trade['stop_loss'])

def render_signal_card(action: str, ticker: str, price: float, conf_pct: float, rules_conf: float, levels: Dict, output: Dict):
    """Render signal card like in the reference image"""
    
    # Asset title
    asset_title = resolve_asset_title_polygon(ticker, ticker)
    
    # AI Override
    ai_override = conf_pct - rules_conf
    
    # Determine card type
    if action == "BUY":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(22, 199, 132, 0.2), rgba(22, 199, 132, 0.05)); 
                    border: 2px solid #16c784; 
                    border-radius: 16px; 
                    padding: 2rem; 
                    margin: 1.5rem 0;">
            <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;">
                Long • Buy Limit
            </div>
            <div style="font-size: 16px; color: #a0a0a0; margin-bottom: 1.5rem;">
                {int(conf_pct)}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"**{asset_title}** • As-of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption(f"AI override: {ai_override:+.0f}% (Rules: {rules_conf:.0f}% → ML: {conf_pct:.0f}%)")
        
    elif action == "SHORT":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(234, 57, 67, 0.2), rgba(234, 57, 67, 0.05)); 
                    border: 2px solid #ea3943; 
                    border-radius: 16px; 
                    padding: 2rem; 
                    margin: 1.5rem 0;">
            <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;">
                Short • Sell Limit
            </div>
            <div style="font-size: 16px; color: #a0a0a0; margin-bottom: 1.5rem;">
                {int(conf_pct)}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"**{asset_title}** • As-of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption(f"AI override: {ai_override:+.0f}% (Rules: {rules_conf:.0f}% → ML: {conf_pct:.0f}%)")
        
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255, 169, 77, 0.2), rgba(255, 169, 77, 0.05)); 
                    border: 2px solid #ffa94d; 
                    border-radius: 16px; 
                    padding: 2rem; 
                    margin: 1.5rem 0;">
            <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;">
                Wait
            </div>
            <div style="font-size: 16px; color: #a0a0a0; margin-bottom: 1.5rem;">
                {int(conf_pct)}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"**{asset_title}** • As-of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption(f"AI override: {ai_override:+.0f}%")
    
    # Price display
    st.markdown(f"### ${price:,.2f}")
    
    # Levels if actionable
    if action in ("BUY", "SHORT"):
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entry", f"${levels['entry']:.2f}")
        with col2:
            st.metric("Stop Loss", f"${levels['sl']:.2f}")
        with col3:
            st.metric("TP1", f"${levels['tp1']:.2f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TP2", f"${levels['tp2']:.2f}")
        with col2:
            st.metric("TP3", f"${levels['tp3']:.2f}")
        
        # R/R
        st.markdown("---")
        rr = rr_line(levels)
        if rr:
            st.info(f"**RR ≈ {rr}**")

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
        st.subheader("Account")
        pnl = user_info['current_capital'] - user_info['initial_capital']
        pnl_pct = (pnl / max(1e-9, user_info['initial_capital'])) * 100
        
        st.metric(
            "Capital", 
            f"${user_info['current_capital']:,.2f}",
            f"{pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )
        
        st.markdown("---")
        
        st.subheader("Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Trades", stats['total_trades'])
            st.metric("Closed", stats['closed_trades'])
        with col2:
            st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
            st.metric("Avg P&L", f"{stats['avg_pnl']:.2f}%")
        
        st.markdown("---")
        
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
            clear_all_caches()
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
    model = st.radio("Select Model", models, horizontal=True, label_visibility="collapsed")
    
    st.write("**Symbol**")
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Enter Symbol", placeholder="aapl", label_visibility="collapsed")
    with col2:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
    
    if analyze_btn:
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
                    
                    rules_conf = float(output.get("rules_confidence", 44.0))
                    render_signal_card(action, ticker.upper(), price, conf_pct, rules_conf, lv, output)
                    
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
                    probs = sig["output"].get('probs') or {}
                    
                    data = {
                        'ticker': sig["ticker"],
                        'direction': 'LONG' if sig["action"] == 'BUY' else 'SHORT',
                        'entry_price': sig["levels"]['entry'],
                        'stop_loss': sig["levels"]['sl'],
                        'tp1': sig["levels"]['tp1'],
                        'tp2': sig["levels"]['tp2'],
                        'tp3': sig["levels"]['tp3'],
                        'tp1_prob': float(probs.get('tp1', 0.0)) * 100 if probs else 0.0,
                        'tp2_prob': float(probs.get('tp2', 0.0)) * 100 if probs else 0.0,
                        'tp3_prob': float(probs.get('tp3', 0.0)) * 100 if probs else 0.0,
                        'confidence': int(sig["confidence"]),
                        'model': sig["model"]
                    }
                    
                    trade_id = db.add_trade(st.session_state.user['user_id'], data, position_pct)
                    st.success(f"Trade #{trade_id} added!")
                    clear_all_caches()
                    del st.session_state["last_signal"]
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                    if ARXORA_DEBUG:
                        st.exception(e)
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
            with st.expander(f"{t['ticker']} — {t['direction']} ({t['remaining_percent']:.0f}%)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entry", f"${t['entry_price']:.2f}")
                with col2:
                    st.metric("Position", f"${t['position_size']:.2f}")
                with col3:
                    st.metric("Confidence", f"{t['confidence']}%")
                
                st.markdown("**TP Status**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"{'✓' if t['tp1_closed'] else '○'} TP1: ${t['take_profit_1']:.2f}")
                with col2:
                    st.write(f"{'✓' if t['tp2_closed'] else '○'} TP2: ${t['take_profit_2']:.2f}")
                with col3:
                    st.write(f"{'✓' if t['tp3_closed'] else '○'} TP3: ${t['take_profit_3']:.2f}")
                
                st.markdown("---")
                
                price = st.number_input("Current Price", float(t['entry_price']), key=f"p_{t['trade_id']}")
                
                tp_level, can_close_tp = get_tp_status(t, price)
                sl_hit = check_sl_hit(t, price)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if tp_level and can_close_tp:
                        if st.button(f"Close {tp_level.upper()}", key=f"{tp_level}_{t['trade_id']}", use_container_width=True):
                            with st.spinner("Closing..."):
                                try:
                                    db.partial_close_trade(t['trade_id'], price, tp_level)
                                    clear_all_caches()
                                    st.success(f"{tp_level.upper()} closed!")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    
                    if sl_hit:
                        st.error("⚠️ SL triggered!")
                        if st.button("Close at SL", key=f"sl_{t['trade_id']}", use_container_width=True):
                            with st.spinner("Closing..."):
                                try:
                                    db.full_close_trade(t['trade_id'], price, "SL_HIT")
                                    clear_all_caches()
                                    st.success("Closed at SL")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                
                with col2:
                    if st.button("Close All", key=f"close_{t['trade_id']}", type="secondary", use_container_width=True):
                        with st.spinner("Closing..."):
                            try:
                                db.full_close_trade(t['trade_id'], price, "MANUAL")
                                clear_all_caches()
                                st.success("Closed!")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

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
        st.dataframe(
            df[['ticker', 'direction', 'entry_price', 'close_price', 'total_pnl_percent', 'close_date']], 
            use_container_width=True
        )
    else:
        st.info("No closed trades")

st.markdown("---")
st.caption("Arxora · Professional Trading Intelligence · 2025")

