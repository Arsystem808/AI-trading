# -*- coding: utf-8 -*-
# app.py — Arxora UI (final EOD): Valid until = конец дня (UTC), примеры тикеров, блок «О проекте» внизу

import os, re, traceback, importlib, sys
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
import streamlit as st

# Опционально
try:
    import pandas as pd  # noqa: F401
except Exception:
    pass

# Лёгкая защита на случай отсутствия filelock
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

# Внешние источники имён (опционально)
try:
    import requests
except Exception:
    requests = None

try:
    import yfinance as yf  # pip install yfinance
except Exception:
    yf = None

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

render_arxora_header()

# ===== Optional performance =====
try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# ===== Helpers =====
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
    if action == "BUY":
        return ("Buy Stop","Entry (Buy Stop)") if entry > last_price else ("Buy Limit","Entry (Buy Limit)")
    else:
        return ("Sell Stop","Entry (Sell Stop)") if entry < last_price else ("Sell Limit","Entry (Sell Limit)"

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

# ===== Имя актива рядом с ценой =====
@st.cache_data(show_spinner=False, ttl=86400)
def resolve_asset_title(raw_symbol: str, normalized: str, out: Optional[Dict[str, Any]]) -> str:
    # 1) имя из модели, если оно уже есть
    if isinstance(out, dict):
        for k in ("asset_name", "symbol_name", "name", "long_name", "display_name", "title"):
            v = out.get(k)
            if isinstance(v, str) and v.strip():
                return f"{v.strip()} ({raw_symbol.upper()})"
    s = (normalized or raw_symbol or "").upper()

    # 2) крипто / форекс / опционы — эвристики без сети
    if s.startswith("X:"):
        pair = s.split(":", 1)[1]
        base = pair.replace("USD", "")
        crypto_names = {"BTC":"Bitcoin","ETH":"Ethereum","SOL":"Solana","BNB":"BNB","DOGE":"Dogecoin","XRP":"XRP"}
        return f"{crypto_names.get(base, base)} / USD ({raw_symbol.upper()})"
    if s.startswith("C:"):
        pair = s.split(":", 1)[1]
        a, b = pair[:3], pair[3:6]
        fx = {"USD":"US Dollar","EUR":"Euro","GBP":"British Pound","JPY":"Japanese Yen","CHF":"Swiss Franc","CAD":"Canadian Dollar","AUD":"Australian Dollar","NZD":"New Zealand Dollar"}
        return f"{fx.get(a,a)} / {fx.get(b,b)} ({raw_symbol.upper()})"
    if s.startswith("O:"):
        tail = s.split(":",1)[1]
        underlying = "".join([ch for ch in tail if ch.isalpha()])
        if underlying:
            return f"{underlying} Option ({raw_symbol.upper()})"

    # 3) yfinance (если установлен)
    if yf is not None:
        try:
            t = yf.Ticker(raw_symbol.upper())
            longname = None
            # new API
            if hasattr(t, "get_info"):
                try: longname = t.get_info().get("longName")
                except Exception: longname = None
            if not longname and hasattr(t, "info"):
                try: longname = t.info.get("longName")
                except Exception: longname = None
            if longname:
                return f"{longname} ({raw_symbol.upper()})"
        except Exception:
            pass

    # 4) Polygon (если есть ключ)
    api = os.getenv("POLYGON_API_KEY") or os.getenv("POLYGON_KEY")
    if api and requests is not None:
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{raw_symbol.upper()}"
            r = requests.get(url, params={"apiKey": api}, timeout=2.5)
            if r.ok:
                name = (r.json() or {}).get("results", {}).get("name")
                if name:
                    return f"{name} ({raw_symbol.upper()})"
        except Exception:
            pass

    # 5) запасной вариант
    return raw_symbol.upper()

# ===== AlphaPulse alias (services.data -> core.data) =====
try:
    import services.data  # noqa
except Exception:
    try:
        import core.data as _core_data
        sys.modules['services.data'] = _core_data
    except Exception:
        pass

# ===== Dynamic import of strategy =====
def _load_strategy_module():
    try:
        mod = importlib.import_module("core.strategy")
        try: mod = importlib.reload(mod)
        except Exception: pass
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

# ===== Стэковая шкала «part‑to‑whole» с легендой =====
def render_confidence_breakdown_inline(ticker: str, conf_pct: float):
    """
    Визуализация:
      - Ширина фиксирована (20 символов), квантизация на 10 сегментов для читабельности.
      - Rules слева, AI справа, оба вписываются внутрь общей уверенности (overall) как части целого.
      - Легенда повторяет порядок сегментов; числовые подписи сохраняются рядом со шкалой.
    """
    st.session_state.setdefault("ai_bar_width", 20)
    st.session_state.setdefault("ai_bar_segments", 10)
    st.session_state.setdefault("ai_bar_order", ["RULES", "AI"])

    try:
        overall = float(conf_pct or 0.0)
    except Exception:
        overall = 0.0
    overall = max(0.0, min(100.0, overall))
    st.session_state["last_overall_conf_pct"] = overall

    rules_pct_in = float(st.session_state.get("last_rules_pct", 44.0))
    ai_delta = overall - rules_pct_in
    ai_pct = max(0.0, min(overall, ai_delta))
    rules_pct = max(0.0, overall - ai_pct)
    sign = "−" if ai_delta < 0 else ""

    WIDTH = int(st.session_state["ai_bar_width"])
    SEGMENTS = max(1, int(st.session_state["ai_bar_segments"]))
    chars_per_segment = max(1, WIDTH // SEGMENTS)

    filled = int(round(WIDTH * (overall / 100.0))) if overall > 0 else 0
    if filled > 0:
        filled = max(chars_per_segment, int(round(filled / chars_per_segment) * chars_per_segment))
        filled = min(WIDTH, filled)

    if overall > 0 and filled > 0:
        rules_chars_f = filled * (rules_pct / overall)
        ai_chars_f = filled * (ai_pct / overall)
    else:
        rules_chars_f = ai_chars_f = 0

    def quantize(x): return int(round(x / chars_per_segment) * chars_per_segment)

    rules_chars = quantize(rules_chars_f)
    ai_chars = quantize(ai_chars_f)
    overflow = (rules_chars + ai_chars) - filled
    if overflow > 0:
        cut_ai = min(ai_chars, overflow)
        ai_chars -= cut_ai
        overflow -= cut_ai
        if overflow > 0:
            rules_chars = max(0, rules_chars - overflow)

    empty_chars = max(0, WIDTH - filled)
    bar = "[" + ("░" * rules_chars) + ("█" * ai_chars) + ("·" * empty_chars) + "]"

    legend_html = f"""
      <div style="display:flex; gap:10px; align-items:center; font-size:12px; opacity:.85; margin-top:6px;">
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:12px; text-align:center;">░</span> Rules
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:12px; text-align:center;">█</span> AI
        </div>
        <div style="margin-left:auto; font-variant-numeric: tabular-nums;">
          Total: {overall:.0f}% • AI: {sign}{ai_pct:.0f}% • Rules: {rules_pct:.0f}%
        </div>
      </div>
    """

    container = f"""
    <div role="group" aria-label="Confidence breakdown"
         style="background:#2b2b2b;color:#fff;border-radius:12px;padding:10px 12px;
                font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;">
      <div>Общая уверенность: {overall:.0f}%</div>
      <div>└ AI override: {sign}{ai_pct:.0f}% {bar}</div>
      {legend_html}
    </div>
    """
    st.markdown(container, unsafe_allow_html=True)

# ===== Main UI =====
st.subheader("AI agents")
models = get_available_models() or ["Octopus"]
model = st.radio("Выберите модель", options=models, index=0, horizontal=False, key="agent_radio")

ticker_input = st.text_input(
    "Тикер",
    placeholder="Примеры ввода: AAPL • SPY • BTCUSD • C:EURUSD • O:SPY240920C500"
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
        if not rec: rec = {"action":"WAIT","confidence":0.0}

        action = str(rec.get("action","WAIT"))
        conf_val = float(rec.get("confidence",0.0))
        conf_pct_val = conf_val*100.0 if conf_val <= 1.0 else conf_val
        st.session_state["last_overall_conf_pct"] = conf_pct_val

        last_price = float(out.get("last_price", 0.0) or 0.0)

        # Человекочитаемое имя актива рядом с ценой
        asset_title = resolve_asset_title(ticker, symbol_for_engine, out)
        st.markdown(
            f"<div style='text-align:center; font-weight:800; letter-spacing:.2px;"
            f"font-size:clamp(20px,3.6vw,34px); margin-top:4px;'>{asset_title}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='font-size:3rem; font-weight:800; text-align:center; "
            f"margin:4px 0 14px 0;'>${last_price:.2f}</div>", unsafe_allow_html=True
        )

        lv = {k: float(out.get("levels", {}).get(k, 0.0)) for k in ("entry","sl","tp1","tp2","tp3")}
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
            <div style="opacity:0.75; font-size:0.95rem; margin-top:2px;">{int(round(conf_pct_val))}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # As‑of / Valid until = конец дня UTC
        now_utc = datetime.now(timezone.utc)
        eod_utc = now_utc.replace(hour=23, minute=59, second=59, microsecond=0)
        st.caption(f"As‑of: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC • Valid until: {eod_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} • Model: {model}")

        # Шкала уверенности
        render_confidence_breakdown_inline(ticker, conf_pct_val)

        # Targets
        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            c1.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            c2.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            c3.markdown(card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp1', 0)*100))}%"), unsafe_allow_html=True)
            c4, c5 = st.columns(2)
            c4.markdown(card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp2', 0)*100))}%"), unsafe_allow_html=True)
            c5.markdown(card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out.get('probs', {}).get('tp3', 0)*100))}%"), unsafe_allow_html=True)
            rr = rr_line(lv)
            if rr:
                st.markdown(f"<div style='margin-top:6px; color:#FFA94D; font-weight:600;'>{rr}</div>", unsafe_allow_html=True)

        # ——— Ваши фразы остаются без изменений ———
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

# ===== Footer / About =====
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
        Octopus-оркестратор взвешивает мнения всех агентов и выдает единый план сделки. Прошлые результаты не гарантируют будущие.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

