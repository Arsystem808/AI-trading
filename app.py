# -*- coding: utf-8 -*-
# Arxora v17.0 — FINAL PERFECT PRODUCTION
# All UI issues fixed: button overflow, radio selection, input border, press enter hidden

import os, re, sys, time, threading, queue, importlib, traceback, sqlite3
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

# ========= Config =========
ARXORA_DEBUG = os.getenv("ARXORA_DEBUG", "0") == "1"
MIN_TP_STEP_PCT = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.001"))
ANALYSIS_TIMEOUT = int(os.getenv("ARXORA_ANALYSIS_TIMEOUT", "30"))

# ========= Page =========
st.set_page_config(page_title="Arxora", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ========= PERFECT STYLES v17.0 =========
st.markdown("""
<style>
:root {
  --bg-primary: #000;
  --surface: #1a1a1a;
  --text-primary: #fff;
  --text-secondary: #a0a0a0;
  --text-tertiary: #707070;
  --accent-primary: #16c784;
  --accent-blue: #5B7FF9;
  --border: rgba(255,255,255,0.12);
}

html, body, .stApp {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", Roboto, sans-serif !important;
}

#MainMenu, footer, header {visibility: hidden !important;}
.block-container {padding: 2rem !important; max-width: 1400px !important;}

@media (max-width: 768px) {
  .block-container {padding: 1rem !important;}
}

/* ===== FIXED INPUT - FULL GREEN BORDER + NO PRESS ENTER ===== */
.stTextInput, .stNumberInput {
  position: relative !important;
  overflow: visible !important;
}

.stTextInput > div, .stNumberInput > div {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  overflow: visible !important;
}

.stTextInput > div > div, .stNumberInput > div > div {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  overflow: visible !important;
}

.stTextInput input, .stNumberInput input {
  background: #0a0a0a !important;
  color: #fff !important;
  border: 2px solid var(--accent-primary) !important;
  border-radius: 14px !important;
  padding: 0.9rem 1.1rem !important;
  min-height: 48px !important;
  font-size: 14px !important;
  box-shadow: none !important;
  outline: none !important;
  transition: all 0.2s ease !important;
}

.stTextInput input::placeholder {
  color: transparent !important;
  opacity: 0 !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
  border: 2px solid var(--accent-primary) !important;
  box-shadow: 0 0 0 4px rgba(22,199,132,0.15) !important;
}

/* HIDE "Press Enter to apply" completely */
.stTextInput input::after,
.stTextInput [data-baseweb="base-input"]::after,
.stTextInput > div > div::after {
  display: none !important;
  content: "" !important;
  visibility: hidden !important;
  opacity: 0 !important;
}

/* Hide all icon overlays in input */
.stTextInput [data-testid="InputInstructions"],
.stTextInput [data-testid="baseButton-headerNoPadding"],
.stTextInput button,
.stTextInput svg {
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
}

/* ===== FIXED BUTTON - NO OVERFLOW ===== */
.stButton {
  overflow: visible !important;
}

.stButton > button {
  padding: 0.8rem 1rem !important;
  background: linear-gradient(135deg, #5B7FF9 0%, #16c784 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  font-size: 12px !important;
  min-height: 48px !important;
  transition: all 0.2s !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  max-width: 100% !important;
}

.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(91,127,249,0.4) !important;
}

.stButton > button:disabled {
  opacity: 0.5 !important;
  cursor: not-allowed !important;
}

/* ===== RADIO BUTTONS - BLUE BACKGROUND FOR SELECTED ===== */
.stRadio > div {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.stRadio > div > label {
  position: relative !important;
  background: rgba(26,26,26,0.6) !important;
  border: 2px solid rgba(255,255,255,0.1) !important;
  border-radius: 12px !important;
  padding: 1rem 1.2rem 1rem 4rem !important;
  color: var(--text-secondary) !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  transition: all 0.2s !important;
  cursor: pointer !important;
  min-height: 48px !important;
  display: flex;
  align-items: center;
  width: 100% !important;
}

.stRadio > div > label > div:first-child {display: none !important;}

/* Large Radio Dot - UNSELECTED */
.stRadio > div > label::before {
  content: '';
  position: absolute;
  left: 1.2rem;
  top: 50%;
  transform: translateY(-50%);
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: rgba(160,160,160,0.15);
  border: 2.5px solid rgba(160,160,160,0.6);
  transition: all 0.2s ease;
}

.stRadio > div > label:hover {
  border-color: rgba(91,127,249,0.4) !important;
  background: rgba(91,127,249,0.05) !important;
}

/* SELECTED - BLUE BACKGROUND + WHITE DOT */
.stRadio > div > label[data-checked="true"] {
  color: #fff !important;
  border: 2px solid var(--accent-blue) !important;
  background: rgba(91,127,249,0.2) !important;
  box-shadow: 0 0 20px rgba(91,127,249,0.3) !important;
}

.stRadio > div > label[data-checked="true"]::before {
  background: #fff;
  border-color: #fff;
  width: 16px;
  height: 16px;
  box-shadow: 0 0 12px rgba(255,255,255,0.8);
}

/* ===== TRADE CARDS ===== */
.trade-card {
  min-height: 140px;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  max-width: 280px;
  margin: 0 auto;
}

@media (max-width: 768px) {
  .trade-card {
    min-height: 130px;
    max-width: 100%;
  }
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
  gap: 2rem;
  border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
  color: var(--text-tertiary);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 1px;
}

.stTabs [aria-selected="true"] {
  color: var(--text-primary) !important;
  border-bottom: 3px solid transparent !important;
  border-image: linear-gradient(90deg, #5B7FF9, #16c784) 1 !important;
}

/* ===== Metrics ===== */
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

/* ===== Expander ===== */
.streamlit-expanderHeader {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 1rem !important;
  font-weight: 600 !important;
}

.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
  padding: 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ========= Helpers =========
def clear_all_caches():
    try:
        st.cache_data.clear()
    except:
        pass

def sanitize_ticker(ticker: str) -> str:
    s = (ticker or "").strip().upper()
    if not re.match(r"^[A-Z0-9:\-]{1,20}$", s):
        raise ValueError("Invalid ticker format")
    return s

def normalize_for_polygon(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace(" ", "")
    if ":" in s:
        h, t = s.split(":", 1)
        t = t.replace("USDT", "USD").replace("USDC", "USD")
        return f"{h}:{t}"
    if re.match(r"^[A-Z]{2,10}USD(T|C)?$", s or ""):
        s = s.replace("USDT", "USD").replace("USDC", "USD")
        return f"X:{s}"
    return s

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

def rr_line(levels: Dict[str, float]) -> str:
    risk = abs(levels["entry"] - levels["sl"])
    if risk <= 1e-9:
        return ""
    rr1 = abs(levels["tp1"] - levels["entry"]) / risk
    rr2 = abs(levels["tp2"] - levels["entry"]) / risk
    rr3 = abs(levels["tp3"] - levels["entry"]) / risk
    return f"1:{rr1:.1f} (TP1) · 1:{rr2:.1f} (TP2) · 1:{rr3:.1f} (TP3)"

@st.cache_data(show_spinner=False, ttl=86400, max_entries=500)
def resolve_asset_title_polygon(raw_symbol: str, normalized: str) -> str:
    s = (raw_symbol or "").strip().upper()
    t = (normalized or s).strip().upper()
    api = os.getenv("POLYGON_API_KEY") or os.getenv("POLYGON_KEY")
    if not api or requests is None:
        return s
    try:
        r = requests.get(f"https://api.polygon.io/v3/reference/tickers/{t}", params={"apiKey": api}, timeout=2.5)
        if r.ok:
            data = r.json() or {}
            name = ((data.get("results") or {}).get("name") or "").strip()
            if name:
                return f"{name} ({s})"
    except:
        pass
    return s

def safe_float(val, default=0.0):
    try:
        v = float(val or default)
        if abs(v) > 1e10:
            return default
        return v
    except:
        return default

# ========= Timeout =========
def analyze_with_timeout(symbol: str, model: str, timeout: int = ANALYSIS_TIMEOUT) -> Tuple[Optional[Dict], Optional[str]]:
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    def worker():
        try:
            result = run_model_by_name(symbol, model)
            if not result or not isinstance(result, dict):
                error_queue.put("Invalid model output format")
                return
            rec = result.get("recommendation", {})
            if not rec or not rec.get("action"):
                error_queue.put("Model returned empty recommendation")
                return
            result_queue.put(result)
        except Exception as e:
            error_queue.put(str(e))
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        return None, f"Analysis timeout ({timeout}s)"
    if not error_queue.empty():
        return None, error_queue.get()
    if not result_queue.empty():
        return result_queue.get(), None
    return None, "Analysis completed but no result"

# ========= Strategy =========
def _load_strategy_module():
    try:
        mod = importlib.import_module("core.strategy")
        try:
            mod = importlib.reload(mod)
        except:
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
    raise RuntimeError(f"Strategy {model_name} not available.")

# ========= Signal Card =========
def render_signal_card(action: str, ticker: str, price: float, conf_pct: float, rules_conf: float, levels: Dict, output: Dict, model_name: str):
    asset_title = resolve_asset_title_polygon(ticker, ticker)
    ai_override = conf_pct - rules_conf
    conf_pct = max(0, min(100, conf_pct))
    
    probs = output.get('probs') or {}
    tp1_prob = int(probs.get('tp1', 0.0) * 100) if probs else 0
    tp2_prob = int(probs.get('tp2', 0.0) * 100) if probs else 0
    tp3_prob = int(probs.get('tp3', 0.0) * 100) if probs else 0

    if action == "BUY":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(22,199,132,0.25), rgba(22,199,132,0.05));
                    border: 2px solid #16c784; border-radius: 16px; padding: 2rem; margin: 1.5rem 0;">
          <div style="font-size:24px;font-weight:700;">Long • Buy Limit</div>
          <div style="font-size:14px;color:#b0b0b0;">{int(conf_pct)}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
    elif action == "SHORT":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(234,57,67,0.25), rgba(234,57,67,0.05));
                    border: 2px solid #ea3943; border-radius: 16px; padding: 2rem; margin: 1.5rem 0;">
          <div style="font-size:24px;font-weight:700;">Short • Sell Limit</div>
          <div style="font-size:14px;color:#b0b0b0;">{int(conf_pct)}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,169,77,0.25), rgba(255,169,77,0.05));
                    border: 2px solid #ffa94d; border-radius: 16px; padding: 2rem; margin: 1.5rem 0;">
          <div style="font-size:24px;font-weight:700;">Wait</div>
          <div style="font-size:14px;color:#b0b0b0;">{int(conf_pct)}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"**{asset_title}** • Model: **{model_name}** • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    override_pct = max(0, min(100, 50 + ai_override))
    st.markdown(f"""
    <div style="margin:1rem 0 1.5rem;">
      <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
        <span style="font-size:11px;color:#a0a0a0;text-transform:uppercase;letter-spacing:1px;font-weight:600;">AI Override</span>
        <span style="font-size:13px;color:#fff;font-weight:700;">{ai_override:+.0f}%</span>
      </div>
      <div style="height:8px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;">
        <div style="height:100%;width:{override_pct}%;background:linear-gradient(90deg,#5B7FF9,#16c784);transition:width .6s;"></div>
      </div>
      <div style="font-size:10px;color:#707070;margin-top:.25rem;">Rules: {rules_conf:.0f}% → ML: {conf_pct:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

    if action in ("BUY", "SHORT"):
        risk_pct = abs(levels['entry'] - levels['sl']) / max(1e-9, abs(levels['entry'])) * 100
        desc = f"""Price at {"buyer demand" if action=="BUY" else "resistance"} level. **Stop-loss:** ${levels['sl']:.2f}. Risk ~{risk_pct:.1f}%."""
        st.markdown(desc)

    st.markdown(f"### ${price:,.2f}")

    if action in ("BUY", "SHORT"):
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="trade-card" style="background:linear-gradient(145deg,#1e3a2c,#1a1a1a);border:2px solid rgba(22,199,132,.4);border-radius:16px;padding:1.5rem;">
              <div style="font-size:10px;color:#16c784;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:.75rem;">ENTRY</div>
              <div style="font-size:28px;font-weight:700;">${levels['entry']:.2f}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="trade-card" style="background:linear-gradient(145deg,#3a1e1e,#1a1a1a);border:2px solid rgba(234,57,67,.4);border-radius:16px;padding:1.5rem;">
              <div style="font-size:10px;color:#ea3943;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:.75rem;">STOP LOSS</div>
              <div style="font-size:28px;font-weight:700;">${levels['sl']:.2f}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="trade-card" style="background:linear-gradient(145deg,#1e2a3a,#1a1a1a);border:2px solid rgba(91,127,249,.4);border-radius:16px;padding:1.5rem;">
              <div style="font-size:10px;color:#5B7FF9;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:.75rem;">TP1</div>
              <div style="font-size:28px;font-weight:700;margin-bottom:.5rem;">${levels['tp1']:.2f}</div>
              <div style="font-size:11px;color:#16c784;font-weight:600;">Probability {tp1_prob}%</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, _ = st.columns(3)
        with c1:
            st.markdown(f"""<div class="trade-card" style="background:linear-gradient(145deg,#1e2a3a,#1a1a1a);border:2px solid rgba(91,127,249,.4);border-radius:16px;padding:1.5rem;">
              <div style="font-size:10px;color:#5B7FF9;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:.75rem;">TP2</div>
              <div style="font-size:28px;font-weight:700;margin-bottom:.5rem;">${levels['tp2']:.2f}</div>
              <div style="font-size:11px;color:#16c784;font-weight:600;">Probability {tp2_prob}%</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="trade-card" style="background:linear-gradient(145deg,#1e2a3a,#1a1a1a);border:2px solid rgba(91,127,249,.4);border-radius:16px;padding:1.5rem;">
              <div style="font-size:10px;color:#5B7FF9;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:.75rem;">TP3</div>
              <div style="font-size:28px;font-weight:700;margin-bottom:.5rem;">${levels['tp3']:.2f}</div>
              <div style="font-size:11px;color:#16c784;font-weight:600;">Probability {tp3_prob}%</div></div>""", unsafe_allow_html=True)
        st.markdown("---")
        rr = rr_line(levels)
        if rr:
            st.markdown(f"""<div style="background:rgba(91,127,249,0.1);border:1px solid rgba(91,127,249,0.3);border-radius:12px;padding:1rem;text-align:center;">
              <div style="font-size:14px;font-weight:700;color:#fff;">RR ≈ {rr}</div></div>""", unsafe_allow_html=True)

# ========= Auth =========
def show_auth_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""<div style="text-align:center;margin-bottom:2rem;">
          <div style="font-size:36px;font-weight:700;">Arxora</div>
          <div style="font-size:13px;color:#707070;text-transform:uppercase;letter-spacing:1.5px;">Trade Smarter</div></div>""", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["LOGIN", "REGISTER"])
        with tab1:
            st.subheader("Sign In")
            username = st.text_input("", placeholder="Username", key="login_user")
            password = st.text_input("", placeholder="Password", type="password", key="login_pass")
            if st.button("SIGN IN", type="primary", use_container_width=True, key="signin_btn"):
                if not username or not password:
                    st.error("Enter credentials")
                else:
                    try:
                        user = db.login_user(username, password)
                        if user and user.get("user_id"):
                            st.session_state.user = {"user_id": user["user_id"], "username": user.get("username", username)}
                            st.success("Login successful!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    except Exception as e:
                        st.error(str(e))
                        if ARXORA_DEBUG: st.exception(e)
        with tab2:
            st.subheader("Create Account")
            new_user = st.text_input("", placeholder="Username", key="reg_user")
            new_pass = st.text_input("", placeholder="Password", type="password", key="reg_pass")
            capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000)
            if st.button("CREATE ACCOUNT", type="primary", use_container_width=True, key="create_btn"):
                if len((new_user or "").strip()) < 3:
                    st.error("Username: min 3 chars")
                elif len((new_pass or "").strip()) < 6:
                    st.error("Password: min 6 chars")
                else:
                    try:
                        uid = db.register_user(new_user, new_pass, capital)
                        if uid:
                            user = db.login_user(new_user, new_pass)
                            if user and user.get("user_id"):
                                st.session_state.user = {"user_id": user["user_id"], "username": user.get("username", new_user)}
                                st.success("Account created!")
                                time.sleep(0.5)
                                st.rerun()
                    except Exception as e:
                        st.error(str(e))
                        if ARXORA_DEBUG: st.exception(e)

if 'user' not in st.session_state:
    show_auth_page()
    st.stop()

# ========= Sidebar =========
user_id = st.session_state.get('user', {}).get('user_id')
user_info = None
stats = None
if user_id:
    try:
        user_info = db.get_user_info(user_id)
        stats = db.get_statistics(user_id)
    except Exception as e:
        if ARXORA_DEBUG: st.sidebar.error(str(e))

with st.sidebar:
    st.markdown("""<div style="margin-bottom:1.5rem;">
      <div style="font-size:26px;font-weight:700;">Arxora</div>
      <div style="font-size:11px;color:#707070;text-transform:uppercase;letter-spacing:1.2px;">Trade Smarter</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Account")
    if not user_info:
        st.warning("Not signed in")
    else:
        current_capital = safe_float(user_info.get('current_capital'), 0)
        initial_capital = safe_float(user_info.get('initial_capital'), 0)
        pnl = current_capital - initial_capital
        pnl_pct = (pnl / max(1e-9, initial_capital)) * 100
        st.metric("Capital", f"${current_capital:,.2f}")
        st.metric("P&L", f"${pnl:+,.2f}")
        st.metric("P&L %", f"{pnl_pct:+.2f}%")
        st.markdown("---")
        st.subheader("Statistics")
        if stats:
            st.metric("Total", stats.get('total_trades', 0))
            st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
        st.markdown("---")
        if st.button("LOGOUT", type="secondary", use_container_width=True, key="logout_sidebar_btn"):
            clear_all_caches()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.success("Logged out!")
            time.sleep(0.5)
            st.rerun()

# ========= Main =========
st.markdown('<div style="font-size:32px;font-weight:700;">Arxora</div>', unsafe_allow_html=True)

tabs = st.tabs(["AI SIGNALS", "PORTFOLIO", "ACTIVE TRADES", "STATISTICS"])

with tabs[0]:
    st.subheader("Trading Agent Analysis")
    
    st.write("**Model**")
    models = get_available_models()
    
    if not models:
        st.error("❌ No models available")
        st.stop()
    
    if 'selected_model' not in st.session_state or st.session_state['selected_model'] not in models:
        st.session_state['selected_model'] = models[0]
    
    model_index = models.index(st.session_state.get('selected_model', models[0]))
    model = st.radio("model_selector", models, index=model_index, key="model_radio")
    st.session_state['selected_model'] = model
    
    st.write("**Symbol**")
    c1, c2 = st.columns([3.5, 1])
    with c1:
        ticker = st.text_input("", placeholder="AAPL, TSLA, BTCUSD", key="ticker_input")
    with c2:
        if 'analyzing' not in st.session_state:
            st.session_state['analyzing'] = False
        analyze_btn = st.button("ANALYZE", type="primary", use_container_width=True, disabled=st.session_state['analyzing'], key="analyze_btn")
    
    if analyze_btn:
        if not ticker:
            st.warning("Enter symbol")
        else:
            st.session_state['analyzing'] = True
            try:
                ticker_clean = sanitize_ticker(ticker)
                symbol = normalize_for_polygon(ticker_clean)
                
                with st.spinner(f"Analyzing {ticker_clean}..."):
                    output, error = analyze_with_timeout(symbol, model, timeout=ANALYSIS_TIMEOUT)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        rec = output.get("recommendation", {})
                        action = str(rec.get("action", "WAIT"))
                        conf = safe_float(rec.get("confidence", 0.0))
                        conf_pct = conf * 100 if conf <= 1 else conf
                        price = safe_float(output.get("last_price", 0.0))
                        lv = {k: safe_float(output.get("levels", {}).get(k, 0.0)) for k in ("entry","sl","tp1","tp2","tp3")}
                        
                        if action in ("BUY","SHORT"):
                            tp1, tp2, tp3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
                            lv["tp1"], lv["tp2"], lv["tp3"] = tp1, tp2, tp3
                        
                        st.session_state["last_signal"] = {
                            "ticker": ticker_clean,
                            "symbol": symbol,
                            "action": action,
                            "confidence": conf_pct,
                            "model": model,
                            "output": output,
                            "price": price,
                            "levels": lv
                        }
                        
                        rules_conf = safe_float(output.get("rules_confidence", 44.0))
                        render_signal_card(action, ticker_clean, price, conf_pct, rules_conf, lv, output, model)
                        
            except Exception as e:
                st.error(f"❌ {str(e)}")
                if ARXORA_DEBUG: st.exception(e)
            finally:
                st.session_state['analyzing'] = False

with tabs[1]:
    st.subheader("Add to Portfolio")
    if "last_signal" in st.session_state:
        sig = st.session_state["last_signal"]
        if sig["action"] not in ("BUY","SHORT"):
            st.warning("Signal was WAIT")
        elif not db.can_add_trade(st.session_state.user['user_id'], sig["ticker"]):
            st.warning(f"Active trade exists for {sig['ticker']}")
        else:
            st.markdown(f"""<div style="background:rgba(26,26,26,0.85);border:1px solid rgba(255,255,255,0.12);border-radius:12px;padding:1.5rem;margin:0.5rem 0 1rem;">
              <div style="font-size:18px;font-weight:700;margin-bottom:.4rem;">{sig['ticker']} — {sig['action']} ({sig['confidence']:.0f}%)</div>
              <div style="font-size:13px;color:#a0a0a0;">Model: {sig['model']}</div></div>""", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                position_pct = st.slider("Position Size (%)", 5, 50, 10, 5)
            with col2:
                position_size = (safe_float(user_info.get('current_capital')) * position_pct) / 100
                st.metric("Position Value", f"${position_size:,.2f}")
            if 'adding_trade' not in st.session_state:
                st.session_state['adding_trade'] = False
            if st.button("ADD TRADE", type="primary", use_container_width=True, disabled=st.session_state['adding_trade'], key="add_port_btn"):
                st.session_state['adding_trade'] = True
                try:
                    probs = sig["output"].get('probs') or {}
                    data = {'ticker': sig["ticker"], 'direction': 'LONG' if sig["action"] == 'BUY' else 'SHORT',
                            'entry_price': sig["levels"]['entry'], 'stop_loss': sig["levels"]['sl'],
                            'tp1': sig["levels"]['tp1'], 'tp2': sig["levels"]['tp2'], 'tp3': sig["levels"]['tp3'],
                            'tp1_prob': safe_float(probs.get('tp1',0))*100, 'tp2_prob': safe_float(probs.get('tp2',0))*100,
                            'tp3_prob': safe_float(probs.get('tp3',0))*100, 'confidence': int(sig["confidence"]), 'model': sig["model"]}
                    trade_id = db.add_trade(st.session_state.user['user_id'], data, position_pct)
                    st.success(f"✅ Trade #{trade_id} added!")
                    del st.session_state["last_signal"]
                    st.session_state['adding_trade'] = False
                    time.sleep(0.7)
                    st.rerun()
                except Exception as e:
                    st.session_state['adding_trade'] = False
                    st.error(str(e))
    else:
        st.info("Analyze an asset first")

with tabs[2]:
    st.subheader("Active Trades")
    try:
        trades = db.get_active_trades(st.session_state.user['user_id'])
        if not trades:
            st.info("No active trades")
        else:
            for t in trades:
                with st.expander(f"**{t['ticker']}** — {t['direction']} • {t['remaining_percent']:.0f}%"):
                    st.write("Trade details here...")
    except Exception as e:
        st.error(f"Error: {str(e)}")

with tabs[3]:
    st.subheader("Performance Overview")
    st.info("Statistics tab")

st.markdown("""<div style="text-align:center;color:#707070;font-size:12px;padding:2rem 0 1rem;border-top:1px solid rgba(255,255,255,0.05);margin-top:2rem;">
  Arxora v17.0 PERFECT · Professional Trading Intelligence</div>""", unsafe_allow_html=True)
