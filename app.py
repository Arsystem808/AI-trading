# -*- coding: utf-8 -*-
# app.py — Arxora UI (final EOD)

import os, re, traceback, importlib, sys, glob, subprocess
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
import streamlit as st
import pandas as pd

# ---- Optional deps guards
try:
    from filelock import FileLock
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

# ---- Page / Branding
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

# ---- Optional performance API stubs (сохраняем совместимость)
try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# ---- Helpers
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

# ---- AlphaPulse alias
try:
    import services.data  # noqa
except Exception:
    try:
        import core.data as _core_data
        sys.modules['services.data'] = _core_data
    except Exception:
        pass

# ---- Strategy import
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

# ---- Confidence UI (обновлённый компактный без Rules)
try:
    from core.ui_confidence import render_confidence_breakdown_inline as _render_breakdown_native
    from core.ui_confidence import get_confidence_breakdown_from_session as _get_conf_from_session
except Exception:
    _render_breakdown_native = None
    _get_conf_from_session = None

def _inject_ai_css_once():
    if st.session_state.get("_ai_css_injected"): 
        return
    st.session_state["_ai_css_injected"] = True
    st.markdown("""
    <style>
      .ai-card{
        background: radial-gradient(120% 160% at 10% 0%, rgba(0,255,255,0.06) 0%, rgba(0,0,0,0) 48%) #0b0f14;
        border: 1px solid rgba(0,255,245,0.18);
        border-radius: 18px;
        padding: 14px 16px 14px 16px;
        box-shadow: 0 0 0 1px rgba(0,255,245,0.05), 0 12px 40px rgba(0,0,0,0.35), inset 0 0 24px rgba(0,255,255,0.03);
      }
      .ai-title{ color:#cfeaf0; font-size:15px; letter-spacing:.2px; margin-bottom:6px; }
      .ai-strong{ color:#19e6f7; font-weight:800; font-size:22px; }
      .ai-sub{ color:#a7bac2; font-size:13px; margin:0 0 8px 0; }
      .ai-meter{
        position: relative;
        width: 100%;
        height: 18px;
        border-radius: 999px;
        background: linear-gradient(180deg,#1a222a,#12171d);
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04);
        overflow: hidden;
      }
      .ai-meter__fill{
        position:absolute; left:0; top:0; bottom:0;
        width: 0%;
        background: linear-gradient(90deg,#22e8ff 0%, #07c5d8 60%, #05a9c0 100%);
        box-shadow: inset 0 -1px 0 rgba(255,255,255,0.25), 0 0 16px rgba(24,232,255,0.35);
      }
      .ai-meter__tail{
        position:absolute; right:0; top:0; bottom:0;
        left: var(--fill, 40%);
        background:
          radial-gradient(circle at 2px 2px, rgba(255,255,255,0.16) 1px, transparent 1.2px) 0 0 / 8px 8px,
          linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        opacity:.7;
      }
      .ai-meter__knob{
        position:absolute; top:50%;
        left: calc(var(--fill, 40%) - 10px);
        width: 20px; height: 20px;
        border-radius: 50%;
        transform: translateY(-50%);
        background: radial-gradient(40% 40% at 50% 50%, #a7f7ff 0%, #22e8ff 60%, #00c7d8 100%);
        box-shadow: 0 0 0 6px rgba(34,232,255,0.16), 0 0 22px rgba(34,232,255,0.45);
        border: 1px solid rgba(255,255,255,0.25);
      }
    </style>
    """, unsafe_allow_html=True)

def render_confidence_breakdown_inline(ticker: str, conf_pct: float):
    # Источник данных
    try:
        data = _get_conf_from_session() if _get_conf_from_session else None
    except Exception:
        data = None
    if not 
        overall = float(conf_pct or 0.0)
        rules = float(st.session_state.get("last_rules_pct", 44.0))
        ai_delta = overall - rules
        data = {
            "overall_confidence_pct": overall,
            "breakdown": {
                "ai_override_delta_pct": ai_delta
            },
            "shap_top": []
        }
    try:
        st.session_state["last_overall_conf_pct"] = float(data["overall_confidence_pct"])
    except Exception:
        pass

    _inject_ai_css_once()
    overall = float(data.get("overall_confidence_pct", 0.0))
    ai_pct = float(data.get("breakdown", {}).get("ai_override_delta_pct", 0.0))
    overall_clamped = max(0.0, min(100.0, overall))
    ai_abs = max(0.0, min(100.0, abs(ai_pct)))
    fill_css = f"{ai_abs:.2f}%"
    sign = "−" if ai_pct < 0 else ""
    ai_text = f"{sign}{ai_abs:.0f}%"

    # Вписываем блок в первую колонку сетки, чтобы ширина совпала с Entry
    col_meter, _ = st.columns([1, 2])
    with col_meter:
        st.markdown(f"""
        <div class="ai-card">
          <div class="ai-title">Общая уверенность: <span class="ai-strong">{overall_clamped:.0f}%</span></div>
          <div class="ai-sub">⟂ AI override: {ai_text}</div>
          <div class="ai-meter" style="--fill:{fill_css}">
            <div class="ai-meter__fill" style="width:{fill_css}"></div>
            <div class="ai-meter__tail"></div>
            <div class="ai-meter__knob"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ---- Data pipeline (оставляем, но без публичного графа)
DATA_DIR = Path("performance_data")
SUMMARY_PATH = Path("performance_summary.csv")
LOCK = FileLock(str(DATA_DIR / ".summary.lock"))

def _seconds_until_eod_utc() -> int:
    now = datetime.now(timezone.utc)
    eod = now.replace(hour=23, minute=59, second=59, microsecond=0)
    return max(5, int((eod - now).total_seconds()))

def _aggregate_performance_to_csv():
    DATA_DIR.mkdir(exist_ok=True)
    frames = []
    for p in DATA_DIR.glob("performance_*_*.csv"):
        try:
            df = pd.read_csv(p, sep=None, engine='python', on_bad_lines='skip')
        except Exception:
            continue
        m = re.match(r"^performance_(.+)_(.+)\.csv$", p.name)
        if m:
            a, t = m.group(1), m.group(2)
            if 'agent' not in df.columns:  df['agent'] = a
            if 'ticker' not in df.columns: df['ticker'] = t
        frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.columns = [c.strip().lower() for c in out.columns]
        out.to_csv(SUMMARY_PATH, index=False)

def _ensure_summary_up_to_date():
    DATA_DIR.mkdir(exist_ok=True)
    with LOCK:
        if os.getenv("ARXORA_AUTO_RUN_BENCHMARK", "0") == "1":
            agents = ["W7","M7","Global","AlphaPulse","Octopus"]
            tickers = ["SPY","QQQ"]
            cmd = ["python3","jobs/daily_benchmarks.py","--agents",*agents,"--tickers",*tickers]
            try:
                subprocess.run(cmd, check=False, capture_output=True)
            except Exception:
                pass
        _aggregate_performance_to_csv()

@st.cache_data(ttl=_seconds_until_eod_utc())
def load_summary_df() -> pd.DataFrame:
    _ensure_summary_up_to_date()
    df = pd.read_csv(SUMMARY_PATH, sep=None, engine='python', on_bad_lines='skip')
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# ---- Main UI
st.subheader("AI agents")
models = get_available_models()
if not models: models = ["Octopus"]
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
        if not rec:
            rec = {"action":"WAIT","confidence":0.0}

        action = str(rec.get("action","WAIT"))
        conf_val = float(rec.get("confidence",0.0))
        conf_pct_val = conf_val*100.0 if conf_val <= 1.0 else conf_val
        st.session_state["last_overall_conf_pct"] = conf_pct_val

        last_price = float(out.get("last_price", 0.0) or 0.0)
        st.markdown(f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>${last_price:.2f}</div>", unsafe_allow_html=True)

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

        # --- Компактный AI override (ширина = колонка как Entry)
        render_confidence_breakdown_inline(ticker, conf_pct_val)

        # --- Targets
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

        # --- Context / Disclaimer
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

        # --- Лёгкий лог (без графиков эффективности)
        try:
            log_agent_performance(model, ticker, datetime.today(), 0.0)
        except Exception:
            pass

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
        st.exception(e)

elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать». Примеры формата показаны в поле ввода.")

# ---- Footer / About
st.markdown("---")
st.markdown("<style>.stButton > button { font-weight: 600; }</style>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="background-color: #000000; color: #ffffff; padding: 15px; border-radius: 10px; margin-top: 6px;">
        <h4 style="font-weight: 600; margin-top: 0;">О проекте</h4>
        <p style="font-weight: 300; margin-bottom: 0;">
        Arxora AI — современное решение, которое помогает трейдерам принимать точные и обоснованные решения
        с помощью ансамбля моделей и калибровки уверенности. Система автоматизирует анализ, повышает качество входов
        и помогает управлять рисками. Несколько ИИ-агентов с разными подходами: трендовые и контртрендовые стратегии. 
        Octopus-оркестратор взвешивает мнения всех агентов и выдает единый план сделки. Прошлые результаты не гарантируют будущие.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
