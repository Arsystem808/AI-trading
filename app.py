# -*- coding: utf-8 -*-
# app.py — устойчивый UI Arxora с динамическим импортом стратегий

import os, re, traceback, importlib
from typing import Any, Dict, Optional
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --------- Перфоманс (мягко) ---------
try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# --------- UI/брендинг ---------
st.set_page_config(page_title="Arxora — трейд-ИИ (MVP)", page_icon="assets/arxora_favicon_512.png", layout="centered")

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

render_arxora_header()

# --------- Хелперы ---------
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010"))
ETF_HINTS  = {"SPY","QQQ","IWM","DIA","EEM","EFA","XLK","XLF","XLE","XLY","XLI","XLV","XLP","XLU","VNQ","GLD","SLV"}
UNIT_STYLE = {"equity":"za_akciyu","etf":"omit","crypto":"per_base","fx":"per_base","option":"per_contract"}

def _fmt(x: Any) -> str:
    try: return f"{float(x):.2f}"
    except Exception: return "0.00"

def compute_risk_pct(levels: Dict[str, float]) -> str:
    entry = float(levels.get("entry", 0)); sl = float(levels.get("sl", 0))
    return "—" if entry == 0 else f"{abs(entry - sl)/max(1e-9,abs(entry))*100.0:.1f}"

def detect_asset_class(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if t.startswith("X:"): return "crypto"
    if t.startswith("C:"): return "fx"
    if t.startswith("O:"): return "option"
    if re.match(r"^[A-Z]{2,10}[-:/]?USD[TDC]?$", t): return "crypto"
    if t in ETF_HINTS: return "etf"
    return "equity"

def parse_base_symbol(ticker: str) -> str:
    t = (ticker or "").upper().replace("X:","").replace("C:","").replace(":","").replace("/","").replace("-","")
    for q in ("USDT","USDC","USD","EUR","JPY","GBP","BTC","ETH"):
        if t.endswith(q) and len(t) > len(q): return t[:-len(q)]
    return re.split(r"[-:/]", (ticker or "").upper())[0].replace("X:","").replace("C:","")

def unit_suffix(ticker: str) -> str:
    style = UNIT_STYLE.get(detect_asset_class(ticker), "omit")
    if style == "za_akciyu":   return " за акцию"
    if style == "per_base":    return f" за 1 {parse_base_symbol(ticker)}"
    if style == "per_contract":return " за контракт"
    return ""

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

def normalize_for_polygon(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace(" ", "")
    if s.startswith(("X:", "C:", "O:")):
        head, tail = s.split(":", 1)
        tail = tail.replace("USDT", "USD").replace("USDC", "USD")
        return f"{head}:{tail}"
    if re.match(r"^[A-Z]{2,10}USD(T|C)?$", s):
        s = s.replace("USDT", "USD").replace("USDC", "USD")
        return f"X:{s}"
    return s

def sanitize_targets(action: str, entry: float, tp1: float, tp2: float, tp3: float):
    step = max(float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010")) * max(1.0, abs(entry)), 1e-6 * max(1.0, abs(entry)))
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
        return "Market price", "Entry (Market)"
    if action == "BUY":
        return ("Buy Stop", "Entry (Buy Stop)") if entry > last_price else ("Buy Limit", "Entry (Buy Limit)")
    else:
        return ("Sell Stop", "Entry (Sell Stop)") if entry < last_price else ("Sell Limit", "Entry (Sell Limit)")

# --------- Confidence breakdown fallback ---------
try:
    from core.ui_confidence import render_confidence_breakdown_inline as _render_breakdown_native
    from core.ui_confidence import get_confidence_breakdown_from_session as _get_conf_from_session
except Exception:
    _render_breakdown_native = None
    _get_conf_from_session = None

def render_confidence_breakdown_inline(ticker: str, conf_pct: float):
    try:
        st.session_state["last_overall_conf_pct"] = float(conf_pct or 0.0)
        st.session_state.setdefault("last_rules_pct", 44.0)
    except Exception:
        pass
    try:
        if _render_breakdown_native:
            return _render_breakdown_native(ticker, float(conf_pct or 0.0))
    except Exception:
        pass
    data = _get_conf_from_session() if _get_conf_from_session else {
        "overall_confidence_pct": float(st.session_state.get("last_overall_conf_pct", conf_pct or 0.0)),
        "breakdown": {
            "rules_pct": float(st.session_state.get("last_rules_pct", 44.0)),
            "ai_override_delta_pct": float(st.session_state.get("last_overall_conf_pct", conf_pct or 0.0)) - float(st.session_state.get("last_rules_pct", 44.0))
        },
        "shap_top": []
    }
    st.markdown("#### Confidence breakdown")
    st.write(f"Общая уверенность: {data.get('overall_confidence_pct',0):.1f}%")
    b = data.get("breakdown", {})
    st.write(f"— Базовые правила: {b.get('rules_pct',0):.1f}%")
    st.write(f"— AI override: {b.get('ai_override_delta_pct',0):.1f}%")

# --------- Динамическая загрузка стратегий ---------
def _load_strategy_module():
    try:
        mod = importlib.import_module("core.strategy")
        return mod, None
    except Exception as e:
        return None, traceback.format_exc()

def run_agent(ticker_norm: str, label: str) -> Dict[str, Any]:
    mod, err = _load_strategy_module()
    if not mod:
        raise RuntimeError("Не удалось импортировать core.strategy:\n" + (err or ""))
    lbl = label.strip().lower()

    # Новый API
    if hasattr(mod, "analyze_by_agent") and hasattr(mod, "Agent"):
        if lbl == "alphapulse": return mod.analyze_by_agent(ticker_norm, mod.Agent.ALPHAPULSE)
        if lbl == "octopus":    return mod.analyze_by_agent(ticker_norm, mod.Agent.OCTOPUS)
        if lbl == "global":     return mod.analyze_by_agent(ticker_norm, mod.Agent.GLOBAL)
        if lbl == "m7pro":      return mod.analyze_by_agent(ticker_norm, mod.Agent.M7PRO)

    # Старый API — прямые функции (надёжнее)
    if lbl == "global" and hasattr(mod, "analyze_asset_global"):
        return mod.analyze_asset_global(ticker_norm, "Краткосрочный")
    if lbl in ("alphapulse","octopus") and hasattr(mod, "analyze_asset_w7"):
        hz = "Среднесрочный" if lbl == "alphapulse" else "Краткосрочный"
        return mod.analyze_asset_w7(ticker_norm, hz)
    if lbl == "m7pro" and hasattr(mod, "analyze_asset_m7"):
        return mod.analyze_asset_m7(ticker_norm, "Краткосрочный", use_ml=True)

    # Универсальный роутер (если есть)
    if hasattr(mod, "analyze_asset"):
        if lbl == "alphapulse": return mod.analyze_asset(ticker_norm, "Среднесрочный", strategy="W7")
        if lbl == "octopus":    return mod.analyze_asset(ticker_norm, "Краткосрочный", strategy="W7")
        if lbl == "global":     return mod.analyze_asset(ticker_norm, "Долгосрочный", strategy="Global")
        if lbl == "m7pro":      return mod.analyze_asset(ticker_norm, "Краткосрочный", strategy="M7")

    # Последний шанс — любая доступная функция
    for fname in ("analyze_asset_m7", "analyze_asset_global", "analyze_asset_w7", "analyze_asset"):
        if hasattr(mod, fname):
            fn = getattr(mod, fname)
            try:
                return fn(ticker_norm)  # сигнатура без параметров горизонта
            except TypeError:
                return fn(ticker_norm, "Краткосрочный")
    raise RuntimeError("Точка входа стратегий отсутствует в core.strategy после импорта.")

# --------- Основной UI ---------
AGENTS = [{"label": "AlphaPulse"}, {"label": "Octopus"}, {"label": "Global"}, {"label": "M7pro"}]
def fmt(i: int) -> str: return AGENTS[i]["label"]
KEY_TICKERS = ["SPY", "QQQ", "BTCUSD", "ETHUSD"]

st.subheader("AI agents")
idx = st.radio("Выберите модель", options=list(range(len(AGENTS))), index=3, format_func=fmt, horizontal=False, key="agent_radio")
agent_rec = AGENTS[idx]

ticker_input = st.text_input("Тикер", value="SPY", placeholder="Примеры: AAPL · TSLA · X:BTCUSD · BTCUSDT", key="main_ticker")
ticker = ticker_input.strip().upper()

def normalize_for_polygon(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace(" ", "")
    if s.startswith(("X:", "C:", "O:")):
        head, tail = s.split(":", 1)
        tail = tail.replace("USDT", "USD").replace("USDC", "USD")
        return f"{head}:{tail}"
    if re.match(r"^[A-Z]{2,10}USD(T|C)?$", s):
        s = s.replace("USDT", "USD").replace("USDC", "USD")
        return f"X:{s}"
    return s

symbol_for_engine = normalize_for_polygon(ticker)
run = st.button("Проанализировать", type="primary", key="main_analyze")
st.write(f"Mode: AI · Model: {agent_rec['label']}")

if run and ticker:
    try:
        out = run_agent(symbol_for_engine, agent_rec["label"])

        # Синхронизация процента для breakdown через Session State
        try:
            conf_val = float(out.get("recommendation", {}).get("confidence", 0.0))
            st.session_state["last_overall_conf_pct"] = float(conf_val * 100.0)
            st.session_state.setdefault("last_rules_pct", 44.0)
        except Exception:
            pass

        last_price = float(out.get("last_price", 0.0))
        st.markdown(f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>${last_price:.2f}</div>", unsafe_allow_html=True)

        # Перфоманс (мягко)
        session_key = f"{agent_rec['label']}_{ticker}_last_price"
        prev_price = st.session_state.get(session_key, None)
        daily_return = 0.0 if (prev_price is None or prev_price == 0) else (last_price - prev_price) / prev_price
        st.session_state[session_key] = last_price
        try: log_agent_performance(agent_rec["label"], ticker, datetime.today(), daily_return)
        except Exception: pass

        # Сигнал/карточка
        action = out["recommendation"]["action"]
        conf = float(out["recommendation"].get("confidence", 0))
        conf_pct = f"{int(round(conf*100))}%"

        lv = dict(out["levels"])
        if action in ("BUY", "SHORT"):
            t1, t2, t3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
            lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

        mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
        header_text = "WAIT"
        if action == "BUY": header_text = f"Long • {mode_text}"
        elif action == "SHORT": header_text = f"Short • {mode_text}"

        st.markdown(f"""
        <div style="background:#c57b0a; padding:14px 16px; border-radius:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:10px;">
            <div style="font-size:1.15rem; font-weight:700;">{header_text}</div>
            <div style="opacity:0.75; font-size:0.95rem; margin-top:2px;">{conf_pct} confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Inline breakdown
        render_confidence_breakdown_inline(ticker, conf*100)

        # Таргеты
        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            with c3: st.markdown(card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp1', 0)*100))}%"), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: st.markdown(card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp2', 0)*100))}%"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp3', 0)*100))}%"), unsafe_allow_html=True)

            rr = rr_line(lv)
            if rr:
                st.markdown(f"<div style='margin-top:6px; color:#FFA94D; font-weight:600;'>{rr}</div>", unsafe_allow_html=True)

        # Перфоманс
        st.subheader(f"Эффективность модели {agent_rec['label']} по ключевым инструментам (3 месяца)")
        cols = st.columns(2)
        for i, tk in enumerate(["SPY","QQQ","BTCUSD","ETHUSD"]):
            perf_data = None
            try: perf_data = get_agent_performance(agent_rec['label'], tk)
            except Exception: perf_data = None
            with cols[i % 2]:
                st.markdown(f"**{tk}**")
                if perf_data is not None:
                    perf_data = perf_data.set_index('date')
                    st.line_chart(perf_data["cumulative_return"])
                else:
                    st.info("Данных пока нет")

        # Контекст/дисклеймер
        CUSTOM_PHRASES = {
            "CONTEXT": {
                "support":["Цена у поддержки. Отложенный вход из зоны спроса, стоп за уровнем."],
                "resistance":["Цена у сопротивления. Отложенный шорт, стоп над уровнем."],
                "neutral":["Баланс. Работаем только по подтверждённому сигналу."]
            },
            "STOPLINE":["Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа."],
            "DISCLAIMER":"AI-анализ не является инвестрекомендацией."
        }
        ctx_key = "support" if action == "BUY" else ("resistance" if action == "SHORT" else "neutral")
        st.markdown(f"<div style='opacity:0.9'>{CUSTOM_PHRASES['CONTEXT'][ctx_key][0]}</div>", unsafe_allow_html=True)
        if action in ("BUY", "SHORT"):
            stopline = CUSTOM_PHRASES["STOPLINE"][0].format(sl=_fmt(lv["sl"]), risk_pct=compute_risk_pct(lv))
            st.markdown(f"<div style='opacity:0.9; margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)
        st.caption(CUSTOM_PHRASES["DISCLAIMER"])

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")

elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать».")

# ---------------- Футер ----------------
st.markdown("---")
st.markdown("""
<style>
    .stButton > button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

with col2:
    if st.button("Arxora", use_container_width=True):
        st.session_state.show_arxora = not st.session_state.get('show_arxora', False)
        st.session_state.show_crypto = False

with col3:
    st.button("US Stocks", use_container_width=True)

with col4:
    if st.button("Crypto", use_container_width=True):
        st.session_state.show_crypto = not st.session_state.get('show_crypto', False)
        st.session_state.show_arxora = False

if st.session_state.get('show_arxora', False):
    st.markdown(
        """
        <div style="background-color: #000000; color: #ffffff; padding: 15px; border-radius: 10px; margin-top: 10px;">
            <h4 style="font-weight: 600;">О проекте</h4>
            <p style="font-weight: 300;">
            Arxora AI — современное решение для трейдинга с ИИ/ML: вероятностная оценка, AI Override и дисциплина риска.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if st.session_state.get('show_crypto', False):
    st.markdown(
        """
        <div style="background-color: #000000; color: #ffffff; padding: 15px; border-radius: 10px; margin-top: 10px;">
            <h4 style="font-weight: 600;">Crypto</h4>
            <p style="font-weight: 300;">
            Анализ криптоактивов с учётом круглосуточной торговли и высокой волатильности.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
