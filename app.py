# -*- coding: utf-8 -*-
# app.py — Arxora UI (final EOD) + User Portfolio Integration + DB Fixes

# Безопасный импорт и вызов ensure_models
try:
    from core.model_fetch import ensure_models
    try:
        ensure_models()  # подтягивает модели в ARXORA_MODEL_DIR или /tmp/models до любых загрузок
    except Exception as e:
        import logging as _lg
        _lg.warning("ensure_models failed: %s", e)
except Exception as e:
    import logging as _lg
    _lg.warning("model_fetch import skipped: %s", e)

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

# ===== НОВОЕ: Импорт системы портфеля =====
try:
    from database import TradingDatabase
    db = TradingDatabase()
except Exception as e:
    st.error(f"⚠️ Не удалось загрузить database.py: {e}")
    st.stop()

try:
    import pandas as pd
except Exception:
    pd = None

# ===== Paths / Env =====
MODEL_DIR = Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))

# Защита от отсутствующей зависимости filelock (без падения UI)
try:
    from filelock import FileLock  # pip install filelock
except Exception:
    class FileLock:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Доп. зависимость для Polygon (необязательна — при отсутствии покажем тикер)
try:
    import requests
except Exception:
    requests = None

# ===== Page / Branding =====
st.set_page_config(page_title="Arxora — трейд‑ИИ (MVP)", page_icon="assets/arxora_favicon_512.png", layout="centered")

def render_arxora_header():
    hero_path = "assets/arxora_logo_hero.png"
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)
    else:
        st.markdown("""
        <div style="border-radius:8px;overflow:hidden;
                    box-shadow:0 0 0 1px rgba(0,0,0,.06),0 12px 32px rgba(0,0,0,.18);">
          <div style="background:#5B5BF7;padding:28px 16px;">
            <div style="max-width:1120px;margin:0 auto;">
              <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
                          color:#fff;font-weight:700;letter-spacing:.4px;
                          font-size:clamp(36px,7vw,72px);line-height:1.05;">
                Arxora
              </div>
            </div>
          </div>
          <div style="background:#000;padding:12px 16px 16px 16px;">
            <div style="max-width:1120px;margin:0 auto;">
              <div style="color:#fff;font-size:clamp(16px,2.4vw,28px);opacity:.92;">trade smarter.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ===== ФИКС ДБ: Диагностика наличия пользователя в текущей БД =====
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
        try:
            if conn:
                conn.close()
        except Exception:
            pass

# ===== НОВОЕ: Аутентификация с фиксами ДБ =====
def show_auth_page():
    render_arxora_header()
    st.title("🔐 Вход в систему")

    # ФИКС ДБ: Показываем активный путь к БД для диагностики
    try:
        st.caption(f"DB path: {Path(db.db_name).resolve()}")
    except Exception:
        pass

    tab1, tab2 = st.tabs(["Вход", "Регистрация"])
    
    with tab1:
        st.subheader("Войти в аккаунт")
        username = st.text_input("Имя пользователя", key="login_username")
        password = st.text_input("Пароль", type="password", key="login_password")
        if st.button("Войти", type="primary"):
            user = db.login_user(username, password)
            if user:
                st.session_state.user = user
                st.success("✅ Успешный вход!")
                st.rerun()
            else:
                st.error("❌ Неверное имя пользователя или пароль")
                # ФИКС ДБ: Повторно показываем путь к БД при ошибке
                try:
                    st.caption(f"DB: {Path(db.db_name).resolve()}")
                except Exception:
                    pass
                if username:
                    # ФИКС ДБ: Проверяем наличие пользователя в текущей БД
                    exists = _user_exists_in_current_db(username)
                    if not exists:
                        st.info("В этой базе такого пользователя нет. Перейдите во вкладку «Регистрация» и создайте его здесь.")
                    else:
                        st.info("Пользователь существует. Проверьте пароль и раскладку/символы (пробелы).")
    
    with tab2:
        st.subheader("Создать аккаунт")
        new_username = st.text_input("Имя пользователя", key="reg_username")
        new_password = st.text_input("Пароль", type="password", key="reg_password")
        initial_capital = st.number_input(
            "Начальный капитал (виртуальный)", 
            min_value=1000, 
            value=10000, 
            step=1000
        )
        if st.button("Зарегистрироваться", type="primary"):
            if len((new_username or "").strip()) < 3:
                st.error("❌ Имя пользователя должно быть минимум 3 символа")
            elif len((new_password or "").strip()) < 6:
                st.error("❌ Пароль должен быть минимум 6 символов")
            else:
                user_id = db.register_user(new_username, new_password, initial_capital)
                if user_id:
                    # ФИКС ДБ: Автоматический вход сразу после регистрации в той же БД
                    user = db.login_user(new_username, new_password)
                    if user:
                        st.session_state.user = user
                        st.success("✅ Аккаунт создан и вход выполнен")
                        st.rerun()
                    else:
                        st.success("✅ Регистрация успешна! Теперь войдите в систему")
                else:
                    st.error("❌ Это имя пользователя уже занято")

# Проверка авторизации
if 'user' not in st.session_state:
    show_auth_page()
    st.stop()

# ===== Sidebar: Информация о пользователе =====
user_info = db.get_user_info(st.session_state.user['user_id'])
stats = db.get_statistics(st.session_state.user['user_id'])

st.sidebar.title(f"👤 {user_info['username']}")
st.sidebar.metric("Текущий капитал", f"${user_info['current_capital']:,.2f}")
st.sidebar.metric("Начальный капитал", f"${user_info['initial_capital']:,.2f}")

pnl_change = user_info['current_capital'] - user_info['initial_capital']
pnl_percent = (pnl_change / max(1e-9, user_info['initial_capital'])) * 100  # ФИКС: Избегаем деления на 0
st.sidebar.metric("Общий P&L", f"${pnl_change:,.2f}", f"{pnl_percent:.2f}%")

st.sidebar.divider()
if st.sidebar.button("🚪 Выйти"):
    del st.session_state.user
    st.rerun()

min_confidence_filter = st.sidebar.slider("Мин. Confidence для добавления", 0, 100, 60)

# ===== Header =====
render_arxora_header()

# ===== Optional performance (оригинальный интерфейс) =====
try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# ===== Helpers (без изменений) =====
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT",  "0.0010"))

def _fmt(x: Any) -> str:
    try: return f"{float(x):.2f}"
    except Exception: return "0.00"

def sanitize_targets(action: str, entry: float, tp1: float, tp2: float, tp3: float):
    step = max(MIN_TP_STEP_PCT * max(1.0, abs(entry)), 1e-6 * max(1.0, abs(entry)))
    if action == "BUY":
        a = sorted([tp1, tp2, tp3]); a[0]=max(a[0], entry+step); a[1]=max(a[1], a[0]+step); a[2]=max(a[2], a[1]+step); return a[0],a[1],a[2]
    if action == "SHORT":
        a = sorted([tp1, tp2, tp3], reverse=True); a[0]=min(a[0], entry-step); a[1]=min(a[1], a[0]-step); a[2]=min(a[2], a[1]-step); return a[0],a[1],a[2]
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
    bg = "#141a20"
    if color == "green": bg = "#006f6f"
    elif color == "red": bg = "#6f0000"
    return f"""
        <div style="background:{bg}; padding:12px 16px; border-radius:14px; border:1px solid rgba(255,255,255,0.06); margin:6px 0;">
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
    if hasattr(mod, fname):
        return getattr(mod, fname)(ticker_norm, "Краткосрочный")
    raise RuntimeError(f"Стратегия {model_name} недоступна.")

def render_confidence_breakdown_inline(ticker: str, conf_pct: float):
    try:
        overall = float(conf_pct or 0.0)
    except Exception:
        overall = 0.0
    st.session_state["last_overall_conf_pct"] = overall
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
    html = f"""
    <div style="background:#2b2b2b;color:#fff;border-radius:12px;padding:10px 12px;
                font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;">
      <div>Общая уверенность: {overall:.0f}%</div>
      <div>└ AI override: {sign}{ai_pct:.0f}% {bar}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ===== ВКЛАДКИ: Сигналы + Портфель + Статистика =====
tab_signals, tab_portfolio, tab_active, tab_stats = st.tabs([
    "🎯 AI Сигналы", 
    "📋 Портфель", 
    "💼 Активные сделки",
    "📈 Статистика"
])

# ===== TAB 1: AI СИГНАЛЫ (ваш оригинальный UI) =====
with tab_signals:
    st.subheader("AI agents")
    models = get_available_models()
    if not models: models = ["Octopus"]
    model = st.radio("Выберите модель", options=models, index=0, horizontal=False, key="agent_radio")

    ticker_input = st.text_input(
        "Тикер",
        placeholder="Примеры ввода: AAPL • SPY • BTCUSD • C:EURUSD"
    )
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
            if not rec:
                rec = {"action":"WAIT","confidence":0.0}

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
                f"<div style='text-align:center; font-weight:800; letter-spacing:.2px; "
                f"font-size:clamp(20px,3.6vw,34px); margin-top:4px;'>{asset_title}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>"
                f"${last_price:.2f}</div>",
                unsafe_allow_html=True
            )

            lv = {k: float(out.get("levels", {}).get(k, 0.0)) for k in ("entry","sl","tp1","tp2","tp3")}
            if action in ("BUY", "SHORT"):
                tp1, tp2, tp3 = lv["tp1"], lv["tp2"], lv["tp3"]
                t1, t2, t3 = sanitize_targets(action, lv["entry"], tp1, tp2, tp3)
                lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

            mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
            header_text = "WAIT"
            if action == "BUY": header_text = f"Long • {mode_text}"
            elif action == "SHORT": header_text = f"Short • {mode_text}"

            st.markdown(f"""
            <div style="background:#c57b0a; padding:14px 16px; border-radius:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:10px;">
                <div style="font-size:1.15rem; font-weight:700;">{header_text}</div>
                <div style="opacity:0.75; font-size:0.95rem; margin-top:2px;">{int(round(conf_pct_val))}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            now_utc = datetime.now(timezone.utc)
            eod_utc = now_utc.replace(hour=23, minute=59, second=59, microsecond=0)
            st.caption(f"As‑of: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC • Valid until: {eod_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} • Model: {model}")

            render_confidence_breakdown_inline(ticker, conf_pct_val)

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
            st.markdown(f"<div style='opacity:0.9'>{CUSTOM_PHRASES['CONTEXT'][ctx_key][0]}</div>", unsafe_allow_html=True)
            if action in ("BUY", "SHORT"):
                stopline = CUSTOM_PHRASES["STOPLINE"][0].format(sl=_fmt(lv["sl"]), risk_pct=f"{abs(lv['entry']-lv['sl'])/max(1e-9,abs(lv['entry']))*100.0:.1f}")
                st.markdown(f"<div style='opacity:0.9; margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)
            st.caption(CUSTOM_PHRASES["DISCLAIMER"])

            try:
                log_agent_performance(model, ticker, datetime.today(), 0.0)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Ошибка анализа: {e}")
            st.exception(e)

    elif not ticker:
        st.info("Введите тикер и нажмите «Проанализировать». Примеры формата показаны в поле ввода.")

# ===== TAB 2: ПОРТФЕЛЬ (добавить сигнал в трекинг) =====
with tab_portfolio:
    st.header("📋 Добавить сигнал в портфель")
    
    if "last_signal" in st.session_state:
        sig = st.session_state["last_signal"]
        ticker = sig["ticker"]
        action = sig["action"]
        conf = sig["confidence"]
        out = sig["output"]
        
        if action not in ("BUY", "SHORT"):
            st.warning("⚠️ Последний сигнал — WAIT. Добавление в портфель недоступно.")
        elif not db.can_add_trade(st.session_state.user['user_id'], ticker):
            st.warning(f"⚠️ По {ticker} уже есть активная сделка! Закройте её перед добавлением новой.")
        else:
            st.success(f"✅ Сигнал: **{ticker}** — **{action}** (Confidence: {conf:.0f}%)")
            
            lv = {k: float(out.get("levels", {}).get(k, 0.0)) for k in ("entry","sl","tp1","tp2","tp3")}
            
            st.write("**Параметры сделки:**")
            st.write(f"- Entry: ${lv['entry']:.2f}")
            st.write(f"- Stop Loss: ${lv['sl']:.2f}")
            st.write(f"- TP1: ${lv['tp1']:.2f} (30% закрытие + SL в безубыток)")
            st.write(f"- TP2: ${lv['tp2']:.2f} (ещё 30%)")
            st.write(f"- TP3: ${lv['tp3']:.2f} (остаток 40%)")
            
            position_percent = st.slider("% от капитала", min_value=5, max_value=50, value=10, step=5)
            position_size = (user_info['current_capital'] * position_percent) / 100
            st.info(f"Размер позиции: **${position_size:,.2f}** ({position_percent}% от капитала)")
            
            potential_profit = position_size * abs(lv['tp1'] - lv['entry']) / max(1e-9, abs(lv['entry']))
            potential_loss = position_size * abs(lv['entry'] - lv['sl']) / max(1e-9, abs(lv['entry']))
            
            col1, col2 = st.columns(2)
            col1.success(f"Потенциальная прибыль (TP1): **${potential_profit:.2f}**")
            col2.error(f"Потенциальный убыток (SL): **${potential_loss:.2f}**")
            
            if conf < min_confidence_filter:
                st.warning(f"⚠️ Confidence ({conf:.0f}%) ниже фильтра ({min_confidence_filter}%). Рекомендуется не добавлять.")
            
            if st.button("✅ ДОБАВИТЬ В ПОРТФЕЛЬ", type="primary", use_container_width=True):
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
                    st.success(f"🎉 Сделка добавлена в портфель! Trade ID: #{trade_id}")
                    st.balloons()
                    del st.session_state["last_signal"]
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    else:
        st.info("📊 Сначала проанализируйте тикер во вкладке 'AI Сигналы', затем добавьте сигнал в портфель здесь.")

# ===== TAB 3: АКТИВНЫЕ СДЕЛКИ =====
with tab_active:
    st.header("💼 Активные сделки")
    active_trades = db.get_active_trades(st.session_state.user['user_id'])
    
    if not active_trades:
        st.info("У вас пока нет активных сделок. Добавьте сигнал во вкладке 'Портфель'!")
    else:
        for trade in active_trades:
            sl_status = "Безубыток" if trade['sl_breakeven'] else f"${trade['stop_loss']:.2f}"
            with st.expander(f"🔹 {trade['ticker']} — {trade['direction']} | Остаток: {trade['remaining_percent']:.0f}% | SL: {sl_status}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entry", f"${trade['entry_price']:.2f}")
                    st.metric("Position", f"${trade['position_size']:.2f}")
                with col2:
                    st.metric("Model", trade['model_used'])
                    st.metric("Confidence", f"{trade['confidence']}%")
                with col3:
                    st.write("**Progress:**")
                    st.write(f"TP1: {'✅' if trade['tp1_closed'] else '⏳'} (30%)")
                    st.write(f"TP2: {'✅' if trade['tp2_closed'] else '⏳'} (30%)")
                    st.write(f"TP3: {'✅' if trade['tp3_closed'] else '⏳'} (40%)")
                
                st.divider()
                current_price = st.number_input(
                    "Текущая цена (для симуляции закрытия)",
                    value=float(trade['entry_price']),
                    key=f"price_{trade['trade_id']}"
                )
                
                # Логика частичного закрытия
                if trade['direction'] == 'LONG':
                    if not trade['tp1_closed'] and current_price >= trade['take_profit_1']:
                        st.success("🎯 TP1 достигнут! Закрыть 30% + перенести SL в безубыток?")
                        if st.button("Закрыть TP1", key=f"tp1_{trade['trade_id']}"):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp1')
                            st.rerun()
                    elif trade['tp1_closed'] and not trade['tp2_closed'] and current_price >= trade['take_profit_2']:
                        st.success("🎯 TP2 достигнут! Закрыть ещё 30%?")
                        if st.button("Закрыть TP2", key=f"tp2_{trade['trade_id']}"):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp2')
                            st.rerun()
                    elif trade['tp2_closed'] and not trade['tp3_closed'] and current_price >= trade['take_profit_3']:
                        st.success("🎯 TP3 достигнут! Закрыть остаток (40%)?")
                        if st.button("Закрыть TP3", key=f"tp3_{trade['trade_id']}"):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp3')
                            st.rerun()
                    elif (trade['sl_breakeven'] and current_price <= trade['entry_price']) or \
                         (not trade['sl_breakeven'] and current_price <= trade['stop_loss']):
                        st.error("⚠️ Stop Loss сработал!")
                        if st.button("Закрыть по SL", key=f"sl_{trade['trade_id']}"):
                            db.full_close_trade(trade['trade_id'], current_price, "SL_HIT")
                            st.rerun()
                elif trade['direction'] == 'SHORT':
                    if not trade['tp1_closed'] and current_price <= trade['take_profit_1']:
                        st.success("🎯 TP1 достигнут! Закрыть 30% + перенести SL в безубыток?")
                        if st.button("Закрыть TP1", key=f"tp1_{trade['trade_id']}"):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp1')
                            st.rerun()
                    elif trade['tp1_closed'] and not trade['tp2_closed'] and current_price <= trade['take_profit_2']:
                        st.success("🎯 TP2 достигнут! Закрыть ещё 30%?")
                        if st.button("Закрыть TP2", key=f"tp2_{trade['trade_id']}"):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp2')
                            st.rerun()
                    elif trade['tp2_closed'] and not trade['tp3_closed'] and current_price <= trade['take_profit_3']:
                        st.success("🎯 TP3 достигнут! Закрыть остаток (40%)?")
                        if st.button("Закрыть TP3", key=f"tp3_{trade['trade_id']}"):
                            db.partial_close_trade(trade['trade_id'], current_price, 'tp3')
                            st.rerun()
                    elif (trade['sl_breakeven'] and current_price >= trade['entry_price']) or \
                         (not trade['sl_breakeven'] and current_price >= trade['stop_loss']):
                        st.error("⚠️ Stop Loss сработал!")
                        if st.button("Закрыть по SL", key=f"sl_{trade['trade_id']}"):
                            db.full_close_trade(trade['trade_id'], current_price, "SL_HIT")
                            st.rerun()
                
                if st.button("🔴 Закрыть всю позицию вручную", key=f"manual_{trade['trade_id']}"):
                    db.full_close_trade(trade['trade_id'], current_price, "MANUAL")
                    st.rerun()

# ===== TAB 4: СТАТИСТИКА =====
with tab_stats:
    st.header("📈 Статистика портфеля")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего сделок", stats['total_trades'])
    col2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    col3.metric("Закрыто", stats['closed_trades'])
    col4.metric("Средний P&L", f"{stats['avg_pnl']:.2f}%")
    
    closed_trades = db.get_closed_trades(st.session_state.user['user_id'])
    
    if closed_trades and pd:
        df = pd.DataFrame(closed_trades)
        df['cumulative_pnl'] = df['total_pnl_dollars'].cumsum()
        df['equity'] = user_info['initial_capital'] + df['cumulative_pnl']
        
        st.subheader("Equity Curve")
        st.line_chart(df.set_index('close_date')['equity'])
        
        st.subheader("История сделок")
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
    else:
        st.info("История сделок пуста. Закройте хотя бы одну сделку для отображения статистики.")

# ===== Footer / About (ваш оригинальный текст) =====
st.markdown("---")
st.markdown("<style>.stButton > button { font-weight: 600; }</style>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="background-color: #000000; color: #ffffff; padding: 15px; border-radius: 10px; margin-top: 6px;">
        <h4 style="font-weight: 600; margin-top: 0;">О проекте</h4>
        <p style="font-weight: 300; margin-bottom: 0;">
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
