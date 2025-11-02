# -*- coding: utf-8 -*-
# app.py — Arxora Trading Platform v15.0 (FINAL PRODUCTION READY)

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
    initial_sidebar_state="expanded"
)

# ========= PRODUCTION-GRADE THEME =========
st.markdown("""
<style>
:root {
    --bg-primary: #000000;
    --bg-secondary: #0a0a0a;
    --surface: #1a1a1a;
    --surface-hover: #252525;
    --accent-primary: #16c784;
    --accent-blue: #5B7FF9;
    --success: #16c784;
    --danger: #ea3943;
    --warning: #ffa94d;
    --text-primary: #ffffff;
    --text-secondary: #a0a0a0;
    --text-tertiary: #707070;
    --border: rgba(255, 255, 255, 0.1);
    --border-light: rgba(255, 255, 255, 0.05);
}

html, body, .stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif !important;
}

#MainMenu, footer, header {visibility: hidden !important;}
.stDeployButton {display: none !important;}

.block-container {
    padding: 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem !important;
    }
    h1 {
        font-size: 24px !important;
    }
    h2 {
        font-size: 18px !important;
    }
}

.element-container {
    margin-bottom: 1rem !important;
}

/* Input fields - clean single border */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1rem !important;
    font-size: 14px !important;
    min-height: 44px !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border: 1px solid var(--accent-primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 1px var(--accent-primary) !important;
}

.stTextInput label, .stNumberInput label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 0.5rem !important;
}

.stTextInput > div,
.stNumberInput > div {
    border: none !important;
    box-shadow: none !important;
}

/* Input instruction hide */
[data-testid="InputInstructions"],
[aria-live="polite"][data-testid="stMarkdownContainer"] {
    display: none !important;
}

/* Buttons */
.stButton > button {
    padding: 0.75rem 1.5rem !important;
    background: var(--accent-primary) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    min-height: 44px !important;
    height: 44px !important;
    width: 100% !important;
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

.stButton > button[kind="secondary"] {
    background: var(--danger) !important;
    color: #fff !important;
}

.stButton > button[kind="secondary"]:hover {
    background: #d32f3a !important;
    box-shadow: 0 4px 12px rgba(234, 57, 67, 0.3) !important;
}

/* Radio - clean with blue dot and production gradient outline */
.stRadio > div {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}

.stRadio > div > label {
    background: transparent !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.25rem 0.65rem 2.5rem !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
    position: relative !important;
}

.stRadio > div > label > div:first-child {
    display: none !important;
}

.stRadio > div > label::before {
    content: '';
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: transparent;
    border: 2px solid var(--text-tertiary);
    transition: all 0.2s;
}

.stRadio > div > label:hover {
    border-color: var(--text-secondary) !important;
    background: var(--surface) !important;
}

.stRadio > div > label[data-checked="true"] {
    background: transparent !important;
    border-color: transparent !important;
    background:
        linear-gradient(var(--bg-primary), var(--bg-primary)) padding-box,
        linear-gradient(135deg, #16c784, #5B7FF9) border-box !important;
    border: 1.5px solid transparent !important;
    color: var(--accent-blue) !important;
    box-shadow: 0 0 0 1px rgba(91,127,249,0.35) inset !important;
}

.stRadio > div > label[data-checked="true"]::before {
    border-color: #5B7FF9;
    background: radial-gradient(circle at center, #5B7FF9 55%, #5B7FF9 55%) !important;
    box-shadow: 0 0 8px rgba(91,127,249,0.6);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-tertiary);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 1px;
    padding: 1rem 0 !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
}

[data-testid="stMetric"] label {
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    font-size: 10px !important;
    letter-spacing: 0.5px !important;
    font-weight: 600 !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* Slider */
.stSlider {
    padding: 0.5rem 0 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    font-weight: 600 !important;
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
    padding: 1.5rem !important;
}

/* Headings */
h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.5px !important;
}

h2 {
    font-size: 20px !important;
    font-weight: 700 !important;
    margin: 1.5rem 0 1rem 0 !important;
}

h3 {
    font-size: 16px !important;
    font-weight: 600 !important;
    margin: 1rem 0 0.5rem 0 !important;
}

h4 {
    font-size: 14px !important;
    font-weight: 700 !important;
    margin: 0.5rem 0 0.25rem 0 !important;
    color: var(--text-primary) !important;
}

/* Captions */
.caption {
    font-size: 12px !important;
    color: var(--text-tertiary) !important;
    line-height: 1.5 !important;
}

/* Dataframe */
.stDataFrame {
    background: var(--surface) !important;
    border-radius: 12px !important;
}

/* Footer */
.footer-text {
    text-align: center;
    color: var(--text-tertiary);
    font-size: 12px;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid var(--border-light);
}

/* Trade card */
.trade-card {
    min-height: 150px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

@media (max-width: 768px) {
    .trade-card {
        min-height: 130px;
    }
}

/* Info boxes */
.stAlert {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
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

def fetch_last_price_polygon(ticker_norm: str) -> Optional[float]:
    """Fetch last price from Polygon API"""
    api = os.getenv("POLYGON_API_KEY") or os.getenv("POLYGON_KEY")
    if not api or requests is None:
        return None
    try:
        if ticker_norm.startswith("X:"):  # crypto
            r = requests.get(
                f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/{ticker_norm}",
                params={"apiKey": api}, timeout=2.5
            )
            if r.ok:
                j = r.json() or {}
                t = (j.get("ticker") or {})
                return float(((t.get("lastTrade") or {}).get("p") or (t.get("day") or {}).get("c") or 0.0))
        else:  # stocks/ETFs
            r = requests.get(
                f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker_norm}",
                params={"apiKey": api}, timeout=2.5
            )
            if r.ok:
                j = r.json() or {}
                t = (j.get("ticker") or {})
                return float(((t.get("lastTrade") or {}).get("p") or (t.get("day") or {}).get("c") or 0.0))
    except Exception:
        return None
    return None

def get_tp_status(trade: Dict, price: float) -> Tuple[Optional[str], bool]:
    """Get which TP can be closed"""
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

def render_signal_card(action: str, ticker: str, price: float, conf_pct: float, rules_conf: float, levels: Dict, output: Dict, model_name: str):
    """Render premium signal card with AI override indicator"""
    
    asset_title = resolve_asset_title_polygon(ticker, ticker)
    ai_override = conf_pct - rules_conf
    
    # Extract probabilities
    probs = output.get('probs') or {}
    tp1_prob = int(probs.get('tp1', 0.0) * 100) if probs else 0
    tp2_prob = int(probs.get('tp2', 0.0) * 100) if probs else 0
    tp3_prob = int(probs.get('tp3', 0.0) * 100) if probs else 0
    
    # Main signal card
    if action == "BUY":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(22, 199, 132, 0.25), rgba(22, 199, 132, 0.05)); 
                    border: 2px solid #16c784; 
                    border-radius: 16px; 
                    padding: 2rem; 
                    margin: 1.5rem 0;
                    box-shadow: 0 8px 24px rgba(22, 199, 132, 0.15);">
            <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; letter-spacing: -0.3px;">
                Long • Buy Limit
            </div>
            <div style="font-size: 14px; color: #b0b0b0;">
                {int(conf_pct)}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    elif action == "SHORT":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(234, 57, 67, 0.25), rgba(234, 57, 67, 0.05)); 
                    border: 2px solid #ea3943; 
                    border-radius: 16px; 
                    padding: 2rem; 
                    margin: 1.5rem 0;
                    box-shadow: 0 8px 24px rgba(234, 57, 67, 0.15);">
            <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; letter-spacing: -0.3px;">
                Short • Sell Limit
            </div>
            <div style="font-size: 14px; color: #b0b0b0;">
                {int(conf_pct)}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255, 169, 77, 0.25), rgba(255, 169, 77, 0.05)); 
                    border: 2px solid #ffa94d; 
                    border-radius: 16px; 
                    padding: 2rem; 
                    margin: 1.5rem 0;
                    box-shadow: 0 8px 24px rgba(255, 169, 77, 0.15);">
            <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; letter-spacing: -0.3px;">
                Wait
            </div>
            <div style="font-size: 14px; color: #b0b0b0;">
                {int(conf_pct)}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show model and asset info
    st.caption(f"**{asset_title}** • Model: **{model_name}** • As-of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    
    # AI Override INDICATOR
    override_pct = min(100, max(0, (ai_override + 50)))
    st.markdown(f"""
    <div style="margin: 1rem 0 1.5rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 11px; color: #a0a0a0; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">AI Override</span>
            <span style="font-size: 13px; color: #ffffff; font-weight: 700;">{ai_override:+.0f}%</span>
        </div>
        <div style="height: 8px; background: rgba(255, 255, 255, 0.05); border-radius: 4px; overflow: hidden;">
            <div style="height: 100%; width: {override_pct}%; background: linear-gradient(90deg, #16c784, #5B7FF9); transition: width 0.6s ease;"></div>
        </div>
        <div style="font-size: 10px; color: #707070; margin-top: 0.25rem; font-family: 'SF Mono', monospace;">
            Rules: {rules_conf:.0f}% → ML: {conf_pct:.0f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Signal Description
    if action in ("BUY", "SHORT"):
        risk_pct = abs(levels['entry'] - levels['sl']) / max(1e-9, abs(levels['entry'])) * 100
        
        if action == "BUY":
            description = f"""
Price at buyer demand level. Optimal entry via AI-analyzed order with growth focus; 
risk control and plan revision essential if consolidation occurs below zone.

**Stop-loss:** ${levels['sl']:.2f}. Potential risk ~{risk_pct:.1f}% from entry.
            """
        else:  # SHORT
            description = f"""
Price at resistance level. Optimal entry via AI-analyzed order with downside focus; 
risk control and plan revision essential if consolidation occurs above zone.

**Stop-loss:** ${levels['sl']:.2f}. Potential risk ~{risk_pct:.1f}% from entry.
            """
        
        st.markdown(description)
    
    st.markdown(f"### ${price:,.2f}")
    
    if action in ("BUY", "SHORT"):
        st.markdown("---")
        
        # Standardized height cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="trade-card" style="background: linear-gradient(145deg, #1e3a2c, #1a1a1a); 
                        border: 2px solid rgba(22, 199, 132, 0.4); 
                        border-radius: 16px; 
                        padding: 1.5rem;
                        box-shadow: 0 8px 16px rgba(22, 199, 132, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);">
                <div style="font-size: 10px; color: #16c784; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.75rem;">ENTRY</div>
                <div style="font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">${levels['entry']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="trade-card" style="background: linear-gradient(145deg, #3a1e1e, #1a1a1a); 
                        border: 2px solid rgba(234, 57, 67, 0.4); 
                        border-radius: 16px; 
                        padding: 1.5rem;
                        box-shadow: 0 8px 16px rgba(234, 57, 67, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);">
                <div style="font-size: 10px; color: #ea3943; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.75rem;">STOP LOSS</div>
                <div style="font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">${levels['sl']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="trade-card" style="background: linear-gradient(145deg, #1e2a3a, #1a1a1a); 
                        border: 2px solid rgba(91, 127, 249, 0.4); 
                        border-radius: 16px; 
                        padding: 1.5rem;
                        box-shadow: 0 8px 16px rgba(91, 127, 249, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);">
                <div style="font-size: 10px; color: #5B7FF9; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.75rem;">TP1</div>
                <div style="font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 0.5rem;">${levels['tp1']:.2f}</div>
                <div style="font-size: 11px; color: #16c784; font-weight: 600;">Probability {tp1_prob}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="trade-card" style="background: linear-gradient(145deg, #1e2a3a, #1a1a1a); 
                        border: 2px solid rgba(91, 127, 249, 0.4); 
                        border-radius: 16px; 
                        padding: 1.5rem;
                        box-shadow: 0 8px 16px rgba(91, 127, 249, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);">
                <div style="font-size: 10px; color: #5B7FF9; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.75rem;">TP2</div>
                <div style="font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 0.5rem;">${levels['tp2']:.2f}</div>
                <div style="font-size: 11px; color: #16c784; font-weight: 600;">Probability {tp2_prob}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="trade-card" style="background: linear-gradient(145deg, #1e2a3a, #1a1a1a); 
                        border: 2px solid rgba(91, 127, 249, 0.4); 
                        border-radius: 16px; 
                        padding: 1.5rem;
                        box-shadow: 0 8px 16px rgba(91, 127, 249, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);">
                <div style="font-size: 10px; color: #5B7FF9; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.75rem;">TP3</div>
                <div style="font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 0.5rem;">${levels['tp3']:.2f}</div>
                <div style="font-size: 11px; color: #16c784; font-weight: 600;">Probability {tp3_prob}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # R/R in WHITE
        rr = rr_line(levels)
        if rr:
            st.markdown(f"""
            <div style="background: rgba(91, 127, 249, 0.1); 
                        border: 1px solid rgba(91, 127, 249, 0.3); 
                        border-radius: 12px; 
                        padding: 1rem;
                        text-align: center;">
                <div style="font-size: 14px; font-weight: 700; color: #ffffff;">RR ≈ {rr}</div>
            </div>
            """, unsafe_allow_html=True)

# ========= Strategy Loading =========
try:
    import services.data
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
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <div style="font-size: 36px; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; letter-spacing: -0.5px;">Arxora</div>
            <div style="font-size: 13px; color: #707070; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500;">Trade Smarter</div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Sign In")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Sign In", type="primary", use_container_width=True):
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
            
            if st.button("Create Account", type="primary", use_container_width=True):
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

# ========= SIDEBAR with Account Window =========
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="font-size: 26px; font-weight: 700; color: #ffffff; margin-bottom: 0.25rem;">Arxora</div>
        <div style="font-size: 11px; color: #707070; text-transform: uppercase; letter-spacing: 1.2px;">Trade Smarter</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    user_info = db.get_user_info(st.session_state.user['user_id'])
    stats = db.get_statistics(st.session_state.user['user_id'])
    
    if user_info:
        # Account Window
        st.subheader("Account")
        
        current_capital = float(user_info['current_capital'])
        initial_capital = float(user_info['initial_capital'])
        pnl = current_capital - initial_capital
        pnl_pct = (pnl / max(1e-9, initial_capital)) * 100
        
        pnl_color = '#16c784' if pnl >= 0 else '#ea3943'
        
        # Account metrics
        st.markdown(f"""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
            <div style="margin-bottom: 1.5rem;">
                <div style="font-size: 10px; color: #707070; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 0.5rem;">Current Capital</div>
                <div style="font-size: 28px; font-weight: 700; color: #ffffff;">${current_capital:,.2f}</div>
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                    <span style="font-size: 12px; color: #a0a0a0;">Initial Capital:</span>
                    <span style="font-size: 12px; color: #ffffff; font-weight: 600;">${initial_capital:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                    <span style="font-size: 12px; color: #a0a0a0;">Total P&L:</span>
                    <span style="font-size: 12px; color: {pnl_color}; font-weight: 600;">${pnl:+,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 12px; color: #a0a0a0;">P&L %:</span>
                    <span style="font-size: 12px; color: {pnl_color}; font-weight: 600;">{pnl_pct:+.2f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", stats['total_trades'])
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
        
        # LOGOUT BUTTON
        if st.button("Logout", use_container_width=True, key="logout_sidebar_btn"):
            clear_all_caches()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ========= MAIN CONTENT =========
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div style="font-size: 32px; font-weight: 700; color: #ffffff; margin-bottom: 0.25rem; letter-spacing: -0.5px;">Arxora</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["AI Signals", "Portfolio", "Active Trades", "Statistics"])

# TAB 1: AI Signals
with tabs[0]:
    st.subheader("Trading Agent Analysis")
    
    st.write("**Model**")
    models = get_available_models()
    model = st.radio("Select Model", models, horizontal=True, label_visibility="collapsed", key="model_radio")
    
    st.write("**Symbol**")
    col1, col2 = st.columns([4, 1])
    with col1:
        ticker = st.text_input("Enter Symbol", placeholder="AAPL, TSLA, BTCUSD, ETHUSD", label_visibility="collapsed")
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
                    render_signal_card(action, ticker.upper(), price, conf_pct, rules_conf, lv, output, model)
                    
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
            st.warning("Signal was WAIT - cannot add to portfolio")
        elif not db.can_add_trade(st.session_state.user['user_id'], sig["ticker"]):
            st.warning(f"Active trade already exists for {sig['ticker']}")
        else:
            # Neutral gray card
            st.markdown(f"""
            <div style="background: rgba(26, 26, 26, 0.8); 
                        border: 1px solid rgba(255, 255, 255, 0.1); 
                        border-radius: 12px; 
                        padding: 1.5rem; 
                        margin: 1rem 0;">
                <div style="font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;">
                    {sig['ticker']} — {sig['action']} ({sig['confidence']:.0f}% confidence)
                </div>
                <div style="font-size: 13px; color: #a0a0a0;">
                    Model: {sig['model']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                position_pct = st.slider("Position Size (%)", 5, 50, 10, 5)
            with col2:
                position_size = (user_info['current_capital'] * position_pct) / 100
                st.metric("Position Value", f"${position_size:,.2f}")
            
            st.markdown("### Trade Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Entry:** ${sig['levels']['entry']:.2f}")
            with col2:
                st.write(f"**Stop Loss:** ${sig['levels']['sl']:.2f}")
            with col3:
                risk_pct = abs(sig['levels']['entry'] - sig['levels']['sl']) / max(1e-9, sig['levels']['entry']) * 100
                st.write(f"**Risk:** {risk_pct:.2f}%")
            
            st.markdown("### Take Profit Levels")
            tp_data = []
            probs = sig["output"].get('probs') or {}
            for i, tp_key in enumerate(['tp1', 'tp2', 'tp3'], 1):
                tp_price = sig['levels'][tp_key]
                tp_prob = int(probs.get(tp_key, 0.0) * 100) if probs else 0
                pnl_pct = abs(tp_price - sig['levels']['entry']) / max(1e-9, sig['levels']['entry']) * 100
                tp_data.append({
                    "Level": f"TP{i}",
                    "Price": f"${tp_price:.2f}",
                    "Probability": f"{tp_prob}%",
                    "Potential P&L": f"{pnl_pct:.2f}%"
                })
            
            if pd:
                st.dataframe(pd.DataFrame(tp_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.markdown("""
            <div style="background: rgba(91, 127, 249, 0.1); 
                        border: 1px solid rgba(91, 127, 249, 0.3); 
                        border-radius: 12px; 
                        padding: 1rem;">
                <div style="font-size: 13px; color: #ffffff; font-weight: 600; margin-bottom: 0.5rem;">Partial Close Strategy</div>
                <div style="font-size: 12px; color: #a0a0a0;">TP1 (50%), TP2 (30%), TP3 (20%). Stop-loss moves to breakeven after TP1.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Add Trade to Portfolio", type="primary", use_container_width=True):
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
                    st.success(f"Trade #{trade_id} added to portfolio!")
                    clear_all_caches()
                    del st.session_state["last_signal"]
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding trade: {str(e)}")
                    if ARXORA_DEBUG:
                        st.exception(e)
    else:
        st.info("Analyze an asset first to add it to your portfolio")

# TAB 3: Active Trades
with tabs[2]:
    st.subheader("Active Trades")

    user_id = st.session_state.user['user_id']
    try:
        active = db.get_active_trades(user_id)
    except Exception as e:
        st.error(f"Failed to load active trades: {e}")
        active = []

    if not active:
        st.info("No active trades at the moment.")
    else:
        for tr in active:
            sym_disp = tr['ticker']
            norm = normalize_for_polygon(sym_disp)
            price = fetch_last_price_polygon(norm) or float(tr['entry_price'])

            next_tp_key, can_close_tp = get_tp_status(tr, price)
            sl_is_hit = check_sl_hit(tr, price)

            rr_levels = {
                "entry": float(tr['entry_price']),
                "sl": float(tr['stop_loss']),
                "tp1": float(tr['take_profit_1']),
                "tp2": float(tr['take_profit_2']),
                "tp3": float(tr['take_profit_3']),
            }
            rr_text = rr_line(rr_levels)

            st.markdown(f"#### {sym_disp} — {tr['direction']}  •  Last: ${price:,.2f}")
            cols = st.columns(4)
            cols[0].metric("Entry", f"${tr['entry_price']:.2f}")
            cols[1].metric("SL", f"${tr['stop_loss']:.2f}")
            cols[2].metric("TP1 | TP2 | TP3", f"${tr['take_profit_1']:.2f} · ${tr['take_profit_2']:.2f} · ${tr['take_profit_3']:.2f}")
            cols[3].metric("RR", rr_text or "—")

            st.markdown("""
            <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">
              <div style="height:100%;width:100%;background:linear-gradient(90deg,#16c784,#5B7FF9);opacity:0.35;"></div>
            </div>
            """, unsafe_allow_html=True)

            b1, b2, b3, b4 = st.columns([1,1,1,1])

            # 1) Close next TP (if threshold met)
            if can_close_tp:
                if b1.button(f"Close {next_tp_key.upper()}", key=f"tpclose_{tr['id']}"):
                    try:
                        db.partial_close(tr['id'], next_tp_key, float(price))
                        if next_tp_key == 'tp1' and not tr.get('sl_breakeven', False):
                            db.move_sl_to_breakeven(tr['id'])
                        st.success(f"{sym_disp}: {next_tp_key.upper()} closed at ${price:,.2f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Partial close failed: {e}")
            else:
                b1.button("Close TP (waiting)", key=f"tpblocked_{tr['id']}", disabled=True)

            # 2) Move SL to Breakeven
            if not tr.get('sl_breakeven', False):
                if b2.button("SL → Breakeven", key=f"slbe_{tr['id']}"):
                    try:
                        db.move_sl_to_breakeven(tr['id'])
                        st.success(f"{sym_disp}: SL moved to breakeven")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to move SL: {e}")
            else:
                b2.button("SL at Breakeven", key=f"slbe_set_{tr['id']}", disabled=True)

            # 3) Market close
            if b3.button("Close at Market", key=f"mktclose_{tr['id']}"):
                try:
                    db.close_trade_market(tr['id'], float(price))
                    st.success(f"{sym_disp}: position closed at market ${price:,.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Market close failed: {e}")

            # 4) Auto SL trigger
            if sl_is_hit:
                if b4.button("Confirm SL Close", key=f"slclose_{tr['id']}"):
                    try:
                        db.close_trade_sl(tr['id'], float(price))
                        st.warning(f"{sym_disp}: stop-loss executed at ${price:,.2f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"SL close failed: {e}")
            else:
                b4.button("SL not hit", key=f"slnoop_{tr['id']}", disabled=True)

            st.markdown("---")

# TAB 4: Statistics
with tabs[3]:
    st.subheader("Performance & Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", stats['total_trades'])
    col2.metric("Closed", stats['closed_trades'])
    col3.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    col4.metric("Avg P&L %", f"{stats['avg_pnl']:.2f}%")
    
    st.markdown("---")
    st.info("Detailed performance charts and historical data will be displayed here.")
