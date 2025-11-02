# -*- coding: utf-8 -*-
# Arxora v18.0 — CLEAN & SIMPLE PRODUCTION
# Back to basics: working UI, no CSS hacks, market + limit orders

import os, re, sys, time, threading, queue, importlib, traceback, sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import streamlit as st

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

ARXORA_DEBUG = os.getenv("ARXORA_DEBUG", "0") == "1"
MIN_TP_STEP_PCT = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.001"))
ANALYSIS_TIMEOUT = int(os.getenv("ARXORA_ANALYSIS_TIMEOUT", "30"))

st.set_page_config(page_title="Arxora", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ========= SIMPLE THEME (CLEAN) =========
st.markdown("""
<style>
:root {
  --bg: #000;
  --surface: #1a1a1a;
  --text: #fff;
  --text-sec: #a0a0a0;
  --text-ter: #707070;
  --accent: #16c784;
  --accent-blue: #5B7FF9;
  --border: rgba(255,255,255,0.12);
}

html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 2rem !important; max-width: 1400px !important; }

/* Simple inputs - just borders */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: #0a0a0a !important;
  border: 1px solid var(--accent) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  padding: 10px 12px !important;
  min-height: 40px !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border: 1px solid var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(22,199,132,0.2) !important;
}

/* Simple buttons */
.stButton > button {
  background: var(--accent-blue) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 10px 16px !important;
  font-weight: 600 !important;
  min-height: 40px !important;
}

.stButton > button:hover { background: #4a6df0 !important; }
.stButton > button:disabled { opacity: 0.5 !important; }

/* Radio - simple */
.stRadio > div { display: flex; flex-direction: column; gap: 0.5rem; }
.stRadio > div > label {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  padding: 8px 12px !important;
  cursor: pointer !important;
}

.stRadio > div > label[data-checked="true"] {
  background: var(--accent-blue) !important;
  border-color: var(--accent-blue) !important;
  color: #fff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] { color: var(--text-ter); }
.stTabs [aria-selected="true"] { color: var(--text) !important; }

[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); padding: 1rem; border-radius: 8px; }

.streamlit-expanderHeader { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
.streamlit-expanderContent { background: var(--surface) !important; border: 1px solid var(--border) !important; }
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
        raise ValueError("Invalid ticker")
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
    return f"1:{rr1:.1f} / 1:{rr2:.1f} / 1:{rr3:.1f}"

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
        return v if abs(v) <= 1e10 else default
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
                error_queue.put("Invalid output")
                return
            rec = result.get("recommendation", {})
            if not rec or not rec.get("action"):
                error_queue.put("Empty recommendation")
                return
            result_queue.put(result)
        except Exception as e:
            error_queue.put(str(e))
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        return None, f"Timeout ({timeout}s)"
    if not error_queue.empty():
        return None, error_queue.get()
    if not result_queue.empty():
        return result_queue.get(), None
    return None, "No result"

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
        raise RuntimeError("Failed to import core.strategy")
    if hasattr(mod, "analyze_asset"):
        return mod.analyze_asset(ticker_norm, "Краткосрочный", model_name)
    reg = getattr(mod, "STRATEGY_REGISTRY", {}) or {}
    if model_name in reg and callable(reg[model_name]):
        return reg[model_name](ticker_norm, "Краткосрочный")
    fname = f"analyze_asset_{model_name.lower()}"
    if hasattr(mod, fname):
        return getattr(mod, fname)(ticker_norm, "Краткосрочный")
    raise RuntimeError(f"Strategy {model_name} not available")

# ========= Signal Card =========
def render_signal_card(action: str, ticker: str, price: float, conf_pct: float, rules_conf: float, levels: Dict, output: Dict, model_name: str):
    asset_title = resolve_asset_title_polygon(ticker, ticker)
    ai_override = conf_pct - rules_conf
    conf_pct = max(0, min(100, conf_pct))
    
    probs = output.get('probs') or {}
    tp1_prob = int(probs.get('tp1', 0.0) * 100) if probs else 0
    tp2_prob = int(probs.get('tp2', 0.0) * 100) if probs else 0
    tp3_prob = int(probs.get('tp3', 0.0) * 100) if probs else 0

    col1, col2 = st.columns([3, 1])
    with col1:
        if action == "BUY":
            st.info(f"🟢 **LONG** • {asset_title} • {int(conf_pct)}% confidence")
        elif action == "SHORT":
            st.error(f"🔴 **SHORT** • {asset_title} • {int(conf_pct)}% confidence")
        else:
            st.warning(f"⏸️ **WAIT** • {asset_title} • {int(conf_pct)}% confidence")
    
    with col2:
        st.metric("AI Override", f"{ai_override:+.0f}%")

    st.caption(f"Model: {model_name} • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entry", f"${levels['entry']:.2f}")
    with col2:
        st.metric("Stop Loss", f"${levels['sl']:.2f}")
    with col3:
        st.metric("Current", f"${price:,.2f}")

    st.write(f"**Price Level:** ${price:,.2f}")

    if action in ("BUY", "SHORT"):
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"TP1 ({tp1_prob}%)", f"${levels['tp1']:.2f}")
        with c2:
            st.metric(f"TP2 ({tp2_prob}%)", f"${levels['tp2']:.2f}")
        with c3:
            st.metric(f"TP3 ({tp3_prob}%)", f"${levels['tp3']:.2f}")
        
        st.write(f"**R/R:** {rr_line(levels)}")

# ========= Auth =========
def show_auth_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("Arxora")
        st.caption("Trade Smarter")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Sign In")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Sign In", type="primary", use_container_width=True):
                if not username or not password:
                    st.error("Enter username and password")
                else:
                    try:
                        user = db.login_user(username, password)
                        if user and user.get("user_id"):
                            st.session_state.user = {"user_id": user["user_id"], "username": user.get("username", username)}
                            st.success("Login successful!")
                            time.sleep(0.3)
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    except Exception as e:
                        st.error(str(e))
        
        with tab2:
            st.subheader("Create Account")
            new_user = st.text_input("Username", key="reg_user")
            new_pass = st.text_input("Password", type="password", key="reg_pass")
            capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000)
            if st.button("Register", type="primary", use_container_width=True):
                if len((new_user or "").strip()) < 3:
                    st.error("Min 3 chars")
                elif len((new_pass or "").strip()) < 6:
                    st.error("Min 6 chars")
                else:
                    try:
                        uid = db.register_user(new_user, new_pass, capital)
                        if uid:
                            user = db.login_user(new_user, new_pass)
                            if user and user.get("user_id"):
                                st.session_state.user = {"user_id": user["user_id"], "username": user.get("username", new_user)}
                                st.success("Created!")
                                time.sleep(0.3)
                                st.rerun()
                    except Exception as e:
                        st.error(str(e))

if 'user' not in st.session_state:
    show_auth_page()
    st.stop()

# ========= Sidebar =========
user_id = st.session_state.get('user', {}).get('user_id')
with st.sidebar:
    st.title("Arxora")
    st.markdown("---")
    
    if user_id:
        user_info = db.get_user_info(user_id)
        stats = db.get_statistics(user_id)
        
        if user_info:
            current = safe_float(user_info.get('current_capital'), 0)
            initial = safe_float(user_info.get('initial_capital'), 0)
            pnl = current - initial
            pnl_pct = (pnl / max(1e-9, initial)) * 100
            
            st.metric("Capital", f"${current:,.0f}")
            st.metric("P&L", f"${pnl:+,.0f}")
            st.metric("P&L %", f"{pnl_pct:+.1f}%")
        
        if stats:
            st.metric("Trades", stats.get('total_trades', 0))
            st.metric("Win Rate", f"{stats.get('win_rate', 0):.0f}%")
        
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ========= Main =========
st.title("Arxora")

tabs = st.tabs(["AI Signals", "Portfolio", "Active Trades", "Statistics"])

with tabs[0]:
    st.subheader("Trading Agent Analysis")
    
    models = get_available_models()
    if not models:
        st.error("No models available")
        st.stop()
    
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = models[0]
    
    st.write("**Select Model:**")
    model = st.radio("Models", models, index=models.index(st.session_state['selected_model']), horizontal=True, key="model_radio")
    st.session_state['selected_model'] = model
    
    st.write("**Enter Symbol:**")
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Symbol", placeholder="AAPL, TSLA, BTCUSD", label_visibility="collapsed", key="ticker_input")
    with col2:
        if 'analyzing' not in st.session_state:
            st.session_state['analyzing'] = False
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True, disabled=st.session_state['analyzing'], key="analyze_btn")
    
    if analyze_btn:
        if not ticker:
            st.warning("Enter symbol")
        else:
            st.session_state['analyzing'] = True
            try:
                ticker_clean = sanitize_ticker(ticker)
                symbol = normalize_for_polygon(ticker_clean)
                
                with st.spinner(f"Analyzing {ticker_clean}..."):
                    output, error = analyze_with_timeout(symbol, model, ANALYSIS_TIMEOUT)
                    
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
            st.warning("Signal was WAIT - cannot add")
        else:
            user_info = db.get_user_info(st.session_state.user['user_id'])
            st.info(f"**{sig['ticker']}** • {sig['action']} • {sig['confidence']:.0f}% • Model: {sig['model']}")
            
            col1, col2 = st.columns(2)
            with col1:
                position_pct = st.slider("Position %", 5, 50, 10)
            with col2:
                pos_size = safe_float(user_info.get('current_capital')) * position_pct / 100
                st.metric("Position $", f"${pos_size:,.0f}")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry", f"${sig['levels']['entry']:.2f}")
            with col2:
                st.metric("Stop Loss", f"${sig['levels']['sl']:.2f}")
            with col3:
                risk = abs(sig['levels']['entry'] - sig['levels']['sl']) / max(1e-9, sig['levels']['entry']) * 100
                st.metric("Risk %", f"{risk:.1f}%")
            
            st.markdown("---")
            
            st.write("**Order Type:**")
            order_type = st.radio("Type", ["Limit", "Market"], horizontal=True, key="order_type_radio")
            
            if 'adding_trade' not in st.session_state:
                st.session_state['adding_trade'] = False
            
            if st.button("Add Trade", type="primary", use_container_width=True, disabled=st.session_state['adding_trade']):
                st.session_state['adding_trade'] = True
                try:
                    probs = sig["output"].get('probs') or {}
                    data = {
                        'ticker': sig["ticker"],
                        'direction': 'LONG' if sig["action"] == 'BUY' else 'SHORT',
                        'order_type': order_type,
                        'entry_price': sig["levels"]['entry'],
                        'stop_loss': sig["levels"]['sl'],
                        'tp1': sig["levels"]['tp1'],
                        'tp2': sig["levels"]['tp2'],
                        'tp3': sig["levels"]['tp3'],
                        'confidence': int(sig["confidence"]),
                        'model': sig["model"]
                    }
                    trade_id = db.add_trade(st.session_state.user['user_id'], data, position_pct)
                    st.success(f"✅ Trade #{trade_id} added! ({order_type})")
                    del st.session_state["last_signal"]
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                finally:
                    st.session_state['adding_trade'] = False
    else:
        st.info("Analyze asset first")

with tabs[2]:
    st.subheader("Active Trades")
    try:
        trades = db.get_active_trades(st.session_state.user['user_id'])
        if not trades:
            st.info("No active trades")
        else:
            for t in trades:
                with st.expander(f"**{t['ticker']}** • {t['direction']} • {t.get('order_type', 'Limit')}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Entry", f"${t['entry_price']:.2f}")
                    with col2: st.metric("Position", f"${t['position_size']:.0f}")
                    with col3: st.metric("Model", t.get('model', 'N/A'))
                    with col4: st.metric("Open", f"{t.get('remaining_percent', 100):.0f}%")
    except Exception as e:
        st.error(f"Error: {e}")

with tabs[3]:
    st.subheader("Statistics")
    try:
        user_id = st.session_state.user['user_id']
        stats = db.get_statistics(user_id)
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total", stats.get('total_trades', 0))
            with col2: st.metric("Closed", stats.get('closed_trades', 0))
            with col3: st.metric("Win Rate", f"{stats.get('win_rate', 0):.0f}%")
            with col4: st.metric("Avg P&L", f"{stats.get('avg_pnl', 0):.2f}%")
        
        closed = db.get_closed_trades(user_id)
        if closed and pd:
            df = pd.DataFrame(closed)
            if not df.empty and 'total_pnl_dollars' in df.columns:
                df['cum_pnl'] = df['total_pnl_dollars'].cumsum()
                st.line_chart(df['cum_pnl'])
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Arxora v18.0 • Trading Intelligence Platform")
