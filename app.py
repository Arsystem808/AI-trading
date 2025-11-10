# -*- coding: utf-8 -*-
# app.py — Arxora UI (production, Saxo-like theme, no auth; сохранены фразы и цвета карточек)

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

# ================= Page config =================
st.set_page_config(page_title="Arxora — трейд‑ИИ (MVP)", page_icon="assets/arxora_favicon_512.png", layout="centered")

# ================= Saxo-like theme (как на скрине) =================
def inject_saxo_theme():
    css = """
    :root{
      --bg:#070708; --panel:#0f1316; --muted:#9aa0a6; --accent:#16b397; --accent-2:#5B5BF7;
      --danger:#ff4c4c; --card-border: rgba(255,255,255,0.06); --amber:#f0b35a;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"]{
      background: linear-gradient(180deg, rgba(7,7,8,1) 0%, rgba(12,12,13,1) 100%) !important;
      color:#e6eef3; font-family:Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .block-container{ max-width:1180px; padding-top:.6rem; }

    /* Header: центр, лаконично */
    .arx-hero{ padding:42px 0 18px 0; text-align:center; }
    .arx-title{ font-weight:800; letter-spacing:.4px; font-size:clamp(28px,6vw,44px); color:#e7edf3; }
    .arx-sub{ margin-top:6px; color:#cfd6dc; font-size:12.5px; letter-spacing:.22em; text-transform:uppercase; }

    /* Tabs: верхний регистр + подчёркивание активной */
    button[data-baseweb="tab"]{
      text-transform:uppercase; letter-spacing:.06em; font-weight:700;
      color:#b7bec6 !important; background:transparent !important; border-radius:8px; padding:10px 14px;
    }
    button[data-baseweb="tab"][aria-selected="true"]{
      color:#e7edf3 !important; position:relative;
    }
    button[data-baseweb="tab"][aria-selected="true"]::after{
      content:""; position:absolute; left:10px; right:10px; bottom:2px; height:3px; border-radius:2px; background:#4a6df0;
    }

    /* Panels */
    .panel{
      background: linear-gradient(180deg, rgba(16,18,20,0.6), rgba(8,9,11,0.6));
      border:1px solid var(--card-border); border-radius:12px; padding:12px; margin-bottom:12px;
      box-shadow: 0 8px 24px rgba(3,4,6,0.55);
    }
    .panel-compact{ padding:10px; border-radius:10px; }

    /* Inputs — «amber editable» */
    input[type="text"], input[type="number"], .stNumberInput input, .stTextInput input{
      background:#14171a; color:#ffb000; border:1px solid #2a323d; border-radius:10px;
    }

    /* Direction header: WAIT amber, LONG/SHORT — ваши цвета */
    .dir-box{ padding:14px 16px; border-radius:16px; margin-bottom:10px; color:#fff; }
    .dir-wait{
      background: radial-gradient(120% 120% at 20% 10%, rgba(240,179,90,0.12) 0%, rgba(240,179,90,0.08) 60%, rgba(240,179,90,0.05) 100%),
                  linear-gradient(180deg, #23180F 0%, #2C1D11 60%, #2F1F12 100%);
      border:1px solid rgba(240,179,90,0.65);
    }
    .dir-long{ background: linear-gradient(98deg, #247102, #247102); border:1px solid rgba(36,113,2,0.55); }
    .dir-short{ background: linear-gradient(98deg, #710224, #710224); border:1px solid rgba(113,2,36,0.55); }

    /* DataFrame */
    .stDataFrame table{ background:transparent; color:#e6eef3; border-collapse:collapse; }
    .stDataFrame th, .stDataFrame td{ padding:8px 10px; border-bottom:1px solid rgba(255,255,255,0.03); }

    .stButton > button{ font-weight:700; border-radius:10px; }
    .muted{ color:#9aa0a6; font-size:13px; }

    /* AI override bar */
    .ai-ovr{
      background:#1a1f26; border:1px solid rgba(255,255,255,.06); border-radius:12px; padding:10px 12px;
      font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_saxo_theme()

# ================= Header =================
def render_arxora_header():
    hero_path = "assets/arxora_logo_hero.png"
    if os.path.exists(hero_path):
        st.image(hero_path, width="stretch")  # вместо use_container_width
    else:
        st.markdown(
            "<div class='arx-hero'><div class='arx-title'>Arxora</div><div class='arx-sub'>TRADE SMARTER</div></div>",
            unsafe_allow_html=True
        )

# ================= Database layer =================
try:
    from database import TradingDatabase
    db = TradingDatabase()
except Exception as e:
    st.error(f"Не удалось загрузить database.py: {e}")
    st.stop()

try: import pandas as pd
except Exception: pd = None

MODEL_DIR = Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
try: import requests
except Exception: requests = None

# ================= Автовход (без аутентификации UI) =================
def _auto_login_or_bootstrap() -> Dict[str, Any]:
    username = os.getenv("ARXORA_AUTO_USER", "demo")
    password = os.getenv("ARXORA_AUTO_PASS", "demo")
    user = None
    try:
        user = db.login_user(username, password)
        if not user:
            try: db.register_user(username, password, 100000.0)
            except Exception: pass
            user = db.login_user(username, password)
    except Exception:
        user = None
    if not user:
        # Страховка: минимальный профиль без БД (дизайн продолжит работать)
        return {'user_id': -1, 'username': username, 'initial_capital': 100000.0, 'current_capital': 100000.0}
    return user

# Сессия пользователя без UI-логина
if 'user' not in st.session_state:
    st.session_state.user = _auto_login_or_bootstrap()

# ================= Sidebar =================
def _load_user_and_stats():
    user_info = {'username':'demo','initial_capital':100000.0,'current_capital':100000.0}
    stats = {'total_trades':0,'win_rate':0.0,'closed_trades':0,'avg_pnl':0.0}
    try:
        if st.session_state.user.get('user_id', -1) != -1:
            user_info = db.get_user_info(st.session_state.user['user_id'])
            stats = db.get_statistics(st.session_state.user['user_id'])
    except Exception:
        pass
    return user_info, stats

user_info, stats = _load_user_and_stats()

st.sidebar.title(user_info['username'])
st.sidebar.metric("Текущий капитал", f"${user_info['current_capital']:,.2f}")
st.sidebar.metric("Начальный капитал", f"${user_info['initial_capital']:,.2f}")

pnl_change = user_info['current_capital'] - user_info['initial_capital']
pnl_percent = (pnl_change / max(1e-9, user_info['initial_capital'])) * 100
st.sidebar.metric("Общий P&L", f"${pnl_change:,.2f}", f"{pnl_percent:.2f}%")

st.sidebar.divider()
min_confidence_filter = st.sidebar.slider("Мин. Confidence для добавления", 0, 100, 60)

# ================= Header render =================
render_arxora_header()

# ================= Optional performance =================
try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# ================= Business helpers =================
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT",  "0.0010"))

def _fmt(x: Any) -> str:
    try: return f"{float(x):.2f}"
    except Exception: return "0.00"

def sanitize_targets(action: str, entry: float, tp1: float, tp2: float, tp3: float):
    step = max(MIN_TP_STEP_PCT * max(1.0, abs(entry)), 1e-6 * max(1.0, abs(entry)))
    if action == "BUY":
        a = sorted([tp1, tp2, tp3])
        a[0]=max(a[0], entry+step); a[1]=max(a[1], a[0]+step); a[2]=max(a[2], a[1]+step)
        return a[0],a[1],a[2]
    if action == "SHORT":
        a = sorted([tp1, tp2, tp3], reverse=True)
        a[0]=min(a[0], entry-step); a[1]=min(a[1], a[0]-step); a[2]=min(a[2], a[1]-step)
        return a[0],a[1],a[2]
    return tp1, tp2, tp3

def entry_mode_labels(action: str, entry: float, last_price: float, eps: float):
    if action not in ("BUY", "SHORT"): return "WAIT", "Entry"
    if abs(entry - last_price) <= eps * max(1.0, abs(last_price)): return "Market price", "Entry (Market)"
    if action == "BUY":  return ("Buy Stop","Entry (Buy Stop)") if entry > last_price else ("Buy Limit","Entry (Buy Limit)")
    else:                return ("Sell Stop","Entry (Sell Stop)") if entry < last_price else ("Sell Limit","Entry (Sell Limit)")

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
    if risk <= 1e-9: return ""
    rr1 = abs(levels["tp1"] - levels["entry"]) / risk
    rr2 = abs(levels["tp2"] - levels["entry"]) / risk
    rr3 = abs(levels["tp3"] - levels["entry"]) / risk
    return f"RR ≈ 1:{rr1:.1f} (TP1) · 1:{rr2:.1f} (TP2) · 1:{rr3:.1f} (TP3)"

def card_html(title: str, value: str, sub: Optional[str]=None, color: Optional[str]=None) -> str:
    # Сохранены ваши цвета карточек
    bg = "#141a20"
    if color == "green": bg = "#247102"
    elif color == "red": bg = "#710224"
    return f"""
        <div class="panel-compact" style="background:{bg}; padding:12px 16px; border-radius:14px; border:1px solid rgba(255,255,255,0.06); margin:6px 0;">
            <div style="font-size:0.9rem; opacity:0.85;">{title}</div>
            <div style="font-size:1.4rem; font-weight:700; margin-top:4px;">{value}</div>
            {f"<div style='font-size:0.8rem; opacity:0.7; margin-top:2px;'>{sub}</div>" if sub else ""}
        </div>
    """

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
            timeout=2.5,
        )
        if r.ok:
            data = r.json() or {}
            name = ((data.get("results") or {}).get("name") or "").strip()
            if name:
                return f"{name} ({s})"
    except Exception:
        pass
    return s

# services.data совместимость
try: import services.data
except Exception:
    try:
        import core.data as _core_data
        sys.modules['services.data'] = _core_data
    except Exception: pass

def _load_strategy_module():
    try:
        mod = importlib.import_module("core.strategy")
        try: mod = importlib.reload(mod)
        except Exception: pass
        return mod, None
    except Exception: return None, traceback.format_exc()

def get_available_models() -> List[str]:
    mod, _ = _load_strategy_module()
    if not mod: return ["Octopus"]
    reg = getattr(mod, "STRATEGY_REGISTRY", {}) or {}
    keys = list(reg.keys())
    return (["Octopus"] if "Octopus" in keys else []) + [k for k in sorted(keys) if k != "Octopus"]

def run_model_by_name(ticker_norm: str, model_name: str) -> Dict[str, Any]:
    mod, err = _load_strategy_module()
    if not mod:
        raise RuntimeError("Не удалось импортировать core.strategy:\n" + (err or ""))
    if hasattr(mod, "analyze_asset"):
        return mod.analyze_asset(ticker_norm, "Краткосрочный", model_name)
    reg = getattr(mod, "STRATEGY_REGISTRY", {}) or {}
    if model_name in reg and callable(reg[model_name]):
        return reg[model_name](ticker_norm, "Краткосрочный")
    fname = f"analyze_asset_{model_name.lower()}"
    if hasattr(mod, fname): return getattr(mod, fname)(ticker_norm, "Краткосрочный")
    raise RuntimeError(f"Стратегия {model_name} недоступна.")

def render_confidence_breakdown_inline(conf_pct: float):
    try: overall = float(conf_pct or 0.0)
    except Exception: overall = 0.0
    rules_pct = float(st.session_state.get("last_rules_pct", 44.0))
    ai_delta = overall - rules_pct
    ai_pct = max(0.0, min(overall, ai_delta))
    sign = "−" if ai_delta < 0 else ""
    WIDTH = 28
    filled = int(round(WIDTH * (overall / 100.0))) if overall > 0 else 0
    ai_chars = int(round(filled * (ai_pct / overall))) if overall > 0 else 0
    rules_chars = max(0, filled - ai_chars)
    empty_chars = max(0, WIDTH - filled)
    bar = "[" + ("░" * rules_chars) + ("█" * ai_chars) + ("·" * empty_chars) + "]"
    st.markdown(f"""
    <div class="ai-ovr">
      <div>Общая уверенность: {overall:.0f}%</div>
      <div>└ AI override: {sign}{ai_pct:.0f}% {bar}</div>
    </div>
    """, unsafe_allow_html=True)

# ================= Tabs =================
tab_signals, tab_portfolio, tab_active, tab_stats = st.tabs([
    "AI SIGNALS", "PORTFOLIO", "ACTIVE TRADES", "STATISTICS"
])

# === TAB 1: AI Сигналы (WAIT‑карточка — amber) ===
with tab_signals:
    st.subheader("AI agents")
    models = get_available_models()
    model = st.radio("Выберите модель", options=models, index=0, horizontal=False, key="agent_radio")

    ticker_input = st.text_input("Тикер", placeholder="Примеры ввода: AAPL • SPY • BTCUSD • C:EURUSD")
    ticker = ticker_input.strip().upper()
    symbol_for_engine = normalize_for_polygon(ticker)

    run = st.button("Проанализировать", type="primary", key="main_analyze")
    st.write(f"Mode: AI · Model: {model}")

    if run and ticker:
        try:
            out = run_model_by_name(symbol_for_engine, model)
            rec = out.get("recommendation")
            if not rec and ("action" in out or "confidence" in out):
                rec = {"action": out.get("action","WAIT"), "confidence": float(out.get("confidence",0.0))}
            if not rec: rec = {"action":"WAIT","confidence":0.0}
            action = str(rec.get("action","WAIT"))
            conf_val = float(rec.get("confidence",0.0))
            conf_pct_val = conf_val*100.0 if conf_val <= 1.0 else conf_val
            st.session_state["last_overall_conf_pct"] = conf_pct_val
            st.session_state["last_signal"] = {
                "ticker": ticker,
                "symbol_for_engine": symbol_for_engine,
                "action": action,
                "confidence": conf_pct_val,
                "model": model,
                "output": out
            }

            last_price = float(out.get("last_price", 0.0) or 0.0)
            asset_title = resolve_asset_title_polygon(ticker, symbol_for_engine)
            st.markdown(
                f"<div style='text-align:center; font-weight:800; letter-spacing:.2px; font-size:clamp(20px,3.6vw,34px); margin-top:4px;'>{asset_title}</div>", 
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>${last_price:.2f}</div>",
                unsafe_allow_html=True
            )

            lv = {k: float(out.get("levels", {}).get(k, 0.0)) for k in ("entry","sl","tp1","tp2","tp3")}
            if action in ("BUY", "SHORT"):
                tp1, tp2, tp3 = lv["tp1"], lv["tp2"], lv["tp3"]
                t1, t2, t3 = sanitize_targets(action, lv["entry"], tp1, tp2, tp3)
                lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

            mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
            # Amber WAIT карточка
            css_class = "dir-wait"; header_text = "WAIT"
            if action == "BUY":
                css_class = "dir-long"; header_text = f"Long • {mode_text}"
            elif action == "SHORT":
                css_class = "dir-short"; header_text = f"Short • {mode_text}"

            st.markdown(
                f"<div class='dir-box {css_class}'><div style='font-size:1.2rem; font-weight:800'>{header_text}</div>"
                f"<div style='opacity:.9; margin-top:2px'>{int(round(conf_pct_val))}% confidence</div></div>",
                unsafe_allow_html=True
            )

            now_utc = datetime.now(timezone.utc)
            eod_utc = now_utc.replace(hour=23, minute=59, second=59, microsecond=0)
            st.caption(f"As‑of: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC • Valid until: {eod_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} • Model: {model}")

            render_confidence_breakdown_inline(conf_pct_val)

            if action in ("BUY", "SHORT"):
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
                with c2: st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
                with c3: st.markdown(card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp1', 0)*100))}%"), unsafe_allow_html=True)
                c4, c5 = st.columns(2)
                with c4: st.markdown(card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp2', 0)*100))}%"), unsafe_allow_html=True)
                with c5: st.markdown(card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp3', 0)*100))}%"), unsafe_allow_html=True)
                rr = rr_line(lv)
                if rr:
                    st.markdown(f"<div style='margin-top:6px; color:#FFA94D; font-weight:600;'>{rr}</div>", unsafe_allow_html=True)

            # Фразы (сохранены)
            CUSTOM_PHRASES = {
                "CONTEXT": {
                    "support":["Цена у уровня покупательской активности. Оптимально — вход по ордеру из AI‑анализа с акцентом на рост; важен контроль риска и пересмотр плана при закреплении ниже зоны."],
                    "resistance":["Риск коррекции повышен. Оптимально — короткий сценарий по ордеру из AI‑анализа; при прорыве и закреплении выше зоны — план пересмотреть."],
                    "neutral":["Баланс. Действовать только по подтверждённому сигналу."]
                },
                "STOPLINE":["Стоп‑лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа."],
                "DISCLAIMER":"AI‑анализ носит информационный характер, не является инвестрекомендацией; рынок меняется быстро, прошлые результаты не гарантируют будущие."
            }
            ctx_key = "support" if action == "BUY" else ("resistance" if action == "SHORT" else "neutral")
            st.markdown(f"<div class='panel'>{CUSTOM_PHRASES['CONTEXT'][ctx_key][0]}</div>", unsafe_allow_html=True)
            if action in ("BUY", "SHORT"):
                stopline = CUSTOM_PHRASES["STOPLINE"][0].format(sl=_fmt(lv["sl"]), risk_pct=f"{abs(lv['entry']-lv['sl'])/max(1e-9,abs(lv['entry']))*100.0:.1f}")
                st.markdown(f"<div class='panel'>{stopline}</div>", unsafe_allow_html=True)
            st.caption(CUSTOM_PHRASES["DISCLAIMER"])

            try: log_agent_performance(model, ticker, datetime.today(), 0.0)
            except Exception: pass

        except Exception as e:
            st.error(f"Ошибка анализа: {e}")
            st.exception(e)
    elif not ticker:
        st.info("Введите тикер и нажмите «Проанализировать». Примеры формата показаны в поле ввода.")

# === TAB 2: Портфель ===
with tab_portfolio:
    st.header("Добавить сигнал в портфель")
    if "last_signal" in st.session_state:
        sig = st.session_state["last_signal"]
        ticker = sig["ticker"]
        action = sig["action"]
        conf = sig["confidence"]
        out = sig["output"]
        if action not in ("BUY", "SHORT"):
            st.warning("Последний сигнал — WAIT. Добавление в портфель недоступно.")
        elif (st.session_state.user.get('user_id', -1) != -1) and (not db.can_add_trade(st.session_state.user['user_id'], ticker)):
            st.warning(f"По {ticker} уже есть активная сделка! Закройте её перед добавлением новой.")
        else:
            st.success(f"Сигнал: {ticker} — {action} (Confidence: {conf:.0f}%)")
            lv = {k: float(out.get("levels", {}).get(k, 0.0)) for k in ("entry","sl","tp1","tp2","tp3")}
            st.write("Параметры сделки:")
            st.write(f"- Entry: ${lv['entry']:.2f}")
            st.write(f"- Stop Loss: ${lv['sl']:.2f}")
            st.write(f"- TP1: ${lv['tp1']:.2f} (30% закрытие + SL в безубыток)")
            st.write(f"- TP2: ${lv['tp2']:.2f} (ещё 30%)")
            st.write(f"- TP3: ${lv['tp3']:.2f} (остаток 40%)")
            position_percent = st.slider("% от капитала", min_value=5, max_value=50, value=10, step=5)
            position_size = (user_info['current_capital'] * position_percent) / 100
            st.info(f"Размер позиции: ${position_size:,.2f} ({position_percent}% от капитала)")
            potential_profit = position_size * abs(lv['tp1'] - lv['entry']) / max(1e-9, abs(lv['entry']))
            potential_loss = position_size * abs(lv['entry'] - lv['sl']) / max(1e-9, abs(lv['entry']))
            col1, col2 = st.columns(2)
            col1.success(f"Потенциальная прибыль (TP1): ${potential_profit:.2f}")
            col2.error(f"Потенциальный убыток (SL): ${potential_loss:.2f}")
            if conf < min_confidence_filter:
                st.warning(f"Confidence ({conf:.0f}%) ниже фильтра ({min_confidence_filter}%). Рекомендуется не добавлять.")
            if st.button("ДОБАВИТЬ В ПОРТФЕЛЬ", type="primary", use_container_width=True):
                if st.session_state.user.get('user_id', -1) == -1:
                    st.error("В демо‑профиле сохранение недоступно. Включите БД или создайте пользователя вручную.")
                else:
                    try:
                        signal_data = {
                            'ticker': ticker,
                            'direction': 'LONG' if action == 'BUY' else 'SHORT',
                            'entry_price': lv['entry'],
                            'stop_loss': lv['sl'],
                            'tp1': lv['tp1'],
                            'tp2': lv['tp2'],
                            'tp3': lv['tp3'],
                            'tp1_prob': float(out.get('probs', {}).get('tp1', 0) * 100),
                            'tp2_prob': float(out.get('probs', {}).get('tp2', 0) * 100),
                            'tp3_prob': float(out.get('probs', {}).get('tp3', 0) * 100),
                            'confidence': int(conf),
                            'model': sig['model']
                        }
                        trade_id = db.add_trade(st.session_state.user['user_id'], signal_data, position_percent)
                        st.success(f"Сделка добавлена в портфель! Trade ID: #{trade_id}")
                        st.balloons()
                        del st.session_state["last_signal"]
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
    else:
        st.info("Сначала проанализируйте тикер во вкладке 'AI Сигналы', затем добавьте сигнал в портфель здесь.")

# === TAB 3: Активные сделки ===
with tab_active:
    st.header("Активные сделки")
    active_trades = []
    try:
        if st.session_state.user.get('user_id', -1) != -1:
            active_trades = db.get_active_trades(st.session_state.user['user_id'])
    except Exception:
        active_trades = []
    if not active_trades:
        st.info("У вас пока нет активных сделок. Добавьте сигнал во вкладке 'Портфель'!")
    else:
        for trade in active_trades:
            sl_status = "Безубыток" if trade['sl_breakeven'] else f"${trade['stop_loss']:.2f}"
            with st.expander(f"{trade['ticker']} — {trade['direction']} | Остаток: {trade['remaining_percent']:.0f}% | SL: {sl_status}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entry", f"${trade['entry_price']:.2f}")
                    st.metric("Position", f"${trade['position_size']:.2f}")
                with col2:
                    st.metric("Model", trade['model_used'])
                    st.metric("Confidence", f"{trade['confidence']}%")
                with col3:
                    st.write("Progress:")
                    st.write(f"TP1: {'✅' if trade['tp1_closed'] else '⏳'} (30%)")
                    st.write(f"TP2: {'✅' if trade['tp2_closed'] else '⏳'} (30%)")
                    st.write(f"TP3: {'✅' if trade['tp3_closed'] else '⏳'} (40%)")
                st.divider()
                current_price = st.number_input(
                    "Текущая цена (для симуляции закрытия)",
                    value=float(trade['entry_price']),
                    key=f"price_{trade['trade_id']}"
                )
                if st.session_state.user.get('user_id', -1) == -1:
                    st.warning("Демо‑профиль: операции закрытия недоступны.")
                else:
                    if trade['direction'] == 'LONG':
                        if not trade['tp1_closed'] and current_price >= trade['take_profit_1']:
                            st.success("TP1 достигнут! Закрыть 30% + перенести SL в безубыток?")
                            if st.button("Закрыть TP1", key=f"tp1_{trade['trade_id']}"):
                                db.partial_close_trade(trade['trade_id'], current_price, 'tp1'); st.rerun()
                        elif trade['tp1_closed'] and not trade['tp2_closed'] and current_price >= trade['take_profit_2']:
                            st.success("TP2 достигнут! Закрыть ещё 30%?")
                            if st.button("Закрыть TP2", key=f"tp2_{trade['trade_id']}"):
                                db.partial_close_trade(trade['trade_id'], current_price, 'tp2'); st.rerun()
                        elif trade['tp2_closed'] and not trade['tp3_closed'] and current_price >= trade['take_profit_3']:
                            st.success("TP3 достигнут! Закрыть остаток (40%)?")
                            if st.button("Закрыть TP3", key=f"tp3_{trade['trade_id']}"):
                                db.partial_close_trade(trade['trade_id'], current_price, 'tp3'); st.rerun()
                        elif (trade['sl_breakeven'] and current_price <= trade['entry_price']) or (not trade['sl_breakeven'] and current_price <= trade['stop_loss']):
                            st.error("Stop Loss сработал!")
                            if st.button("Закрыть по SL", key=f"sl_{trade['trade_id']}"):
                                db.full_close_trade(trade['trade_id'], current_price, "SL_HIT"); st.rerun()
                    elif trade['direction'] == 'SHORT':
                        if not trade['tp1_closed'] and current_price <= trade['take_profit_1']:
                            st.success("TP1 достигнут! Закрыть 30% + перенести SL в безубыток?")
                            if st.button("Закрыть TP1", key=f"tp1_{trade['trade_id']}"):
                                db.partial_close_trade(trade['trade_id'], current_price, 'tp1'); st.rerun()
                        elif trade['tp1_closed'] and not trade['tp2_closed'] and current_price <= trade['take_profit_2']:
                            st.success("TP2 достигнут! Закрыть ещё 30%?")
                            if st.button("Закрыть TP2", key=f"tp2_{trade['trade_id']}"):
                                db.partial_close_trade(trade['trade_id'], current_price, 'tp2'); st.rерun()
                        elif trade['tp2_closed'] and not trade['tp3_closed'] and current_price <= trade['take_profit_3']:
                            st.success("TP3 достигнут! Закрыть остаток (40%)?")
                            if st.button("Закрыть TP3", key=f"tp3_{trade['trade_id']}"):
                                db.partial_close_trade(trade['trade_id'], current_price, 'tp3'); st.rerun()
                        elif (trade['sl_breakeven'] and current_price >= trade['entry_price']) or (not trade['sl_breakeven'] and current_price >= trade['stop_loss']):
                            st.error("Stop Loss сработал!")
                            if st.button("Закрыть по SL", key=f"sl_{trade['trade_id']}"):
                                db.full_close_trade(trade['trade_id'], current_price, "SL_HIT"); st.rerun()

# === TAB 4: Статистика ===
with tab_stats:
    st.header("Статистика портфеля")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего сделок", stats['total_trades'])
    col2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    col3.metric("Закрыто", stats['closed_trades'])
    col4.metric("Средний P&L", f"{stats['avg_pnl']:.2f}%")
    closed_trades = []
    try:
        if st.session_state.user.get('user_id', -1) != -1:
            closed_trades = db.get_closed_trades(st.session_state.user['user_id'])
    except Exception:
        closed_trades = []
    if closed_trades and pd:
        df = pd.DataFrame(closed_trades)
        df['cumulative_pnl'] = df['total_pnl_dollars'].cumsum()
        df['equity'] = user_info['initial_capital'] + df['cumulative_pnl']
        st.subheader("Equity Curve")
        try:
            st.line_chart(df.set_index('close_date')['equity'])
        except Exception:
            st.line_chart(df['equity'])
        st.subheader("История сделок")
        try:
            st.dataframe(
                df[[
                    'ticker', 'direction', 'entry_price', 'close_price',
                    'close_reason', 'total_pnl_percent', 'total_pnl_dollars', 'close_date'
                ]].style.format({
                    'entry_price': '${:.2f}',
                    'close_price': '${:.2f}',
                    'total_pnl_percent': '{:.2f}%',
                    'total_pnl_dollars': '${:.2f}'
                }),
                use_container_width=True
            )
        except Exception:
            st.dataframe(df, use_container_width=True)
    else:
        st.info("История сделок пуста. Закройте хотя бы одну сделку для отображения статистики.")

# ================= Footer =================
st.markdown("---")
st.markdown(
    """
    <div class="panel">
        <h4 style="font-weight: 600; margin: 0 0 8px 0;">О проекте</h4>
        <p class="muted" style="margin:0;">
        Arxora — современное решение, которое помогает трейдерам принимать точные и обоснованные решения
        с помощью ансамбля моделей и калибровки уверенности. Система автоматизирует анализ, повышает качество входов
        и помогает управлять рисками. Несколько ИИ-агентов с разными подходами: трендовые и контртрендовые стратегии. 
        Octopus-оркестратор взвешивает мнения всех агентов и выдает единый план сделки. 
        AI Override — это встроенный механизм, который позволяет искусственному интеллекту вмешиваться в работу базовых алгоритмов и принимать более точные решения в моменты, когда рынок ведёт себя нестандартно.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
