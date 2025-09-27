# -*- coding: utf-8 -*-
# app.py — устойчивый UI Arxora: все модели из реестра, корректный показ note_html+context, фильтрация внутренних подробностей, alias для AlphaPulse

import os, re, traceback, importlib, sys
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

# ===== Optional performance (safe imports) =====
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

def compute_risk_pct(levels: Dict[str, float]) -> str:
    entry = float(levels.get("entry", 0)); sl = float(levels.get("sl", 0))
    return "—" if entry == 0 else f"{abs(entry - sl)/max(1e-9,abs(entry))*100.0:.1f}"

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

# ===== AlphaPulse import compatibility alias =====
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
        try:
            mod = importlib.reload(mod)
        except Exception:
            pass
        return mod, None
    except Exception as e:
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
    # Универсальный роутер — предпочтительно
    if hasattr(mod, "analyze_asset"):
        return mod.analyze_asset(ticker_norm, "Краткосрочный", model_name)
    # Реестр как запасной вариант
    reg = getattr(mod, "STRATEGY_REGISTRY", {}) or {}
    if model_name in reg and callable(reg[model_name]):
        return reg[model_name](ticker_norm, "Краткосрочный")
    # Прямые функции
    fname = f"analyze_asset_{model_name.lower()}"
    if hasattr(mod, fname):
        return getattr(mod, fname)(ticker_norm, "Краткосрочный")
    raise RuntimeError(f"Стратегия {model_name} недоступна.")

# ===== Confidence breakdown (fallback) =====
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

# ===== Internal text filter for user‑facing UI =====
def _is_internal_line(s: str) -> bool:
    if not isinstance(s, str): return False
    s_low = s.lower()
    bad_keys = [
        "orchestrated", "global=", "m7=", "w7=", "alphapulse=",
        "fib_", "на уровне", "level", "pivot"
    ]
    return any(k in s_low for k in bad_keys)

def _sanitize_html(html: str) -> Optional[str]:
    if not isinstance(html, str) or not html.strip():
        return None
    if _is_internal_line(html):
        return None
    return html

# ===== Main UI =====
st.subheader("AI agents")
models = get_available_models()
if not models: models = ["Octopus"]
model = st.radio("Выберите модель", options=models, index=0, horizontal=False, key="agent_radio")

ticker_input = st.text_input("Тикер", value="SPY", placeholder="Примеры: AAPL · TSLA · X:BTCUSD · BTCUSDT · C:EURUSD · O:SPY240920C500")
ticker = ticker_input.strip().upper()
symbol_for_engine = normalize_for_polygon(ticker)

run = st.button("Проанализировать", type="primary", key="main_analyze")
st.write(f"Mode: AI · Model: {model}")

if run and ticker:
    try:
        out = run_model_by_name(symbol_for_engine, model)

        # Совместимость: если нет recommendation — собрать из верхнего уровня
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

        # As-of / Valid
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        ttl_h = int(os.getenv("ARXORA_TTL_HOURS", "24"))
        valid_until = (datetime.utcnow() + timedelta(hours=ttl_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        st.caption(f"As‑of: {now_iso} UTC • Valid until: {valid_until} • Model: {model}")

        # Пояснения: сначала note_html (если не внутренний), затем внешний context
        note_html = _sanitize_html(out.get("note_html", ""))
        if note_html:
            st.markdown(note_html, unsafe_allow_html=True)
        ctx = out.get("context", [])
        if isinstance(ctx, list) and ctx:
            for line in ctx:
                if not _is_internal_line(str(line)):
                    st.caption(str(line))
        elif isinstance(ctx, str) and ctx and not _is_internal_line(ctx):
            st.caption(ctx)

        # Breakdown
        render_confidence_breakdown_inline(ticker, conf_pct_val)

        # Таргеты
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

        # Лёгкий перфоманс (если доступен)
        try:
            log_agent_performance(model, ticker, datetime.today(), 0.0)
        except Exception:
            pass

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
        st.exception(e)

elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать». Примеры формата показаны в placeholder.")

# ===== Footer / Info toggles =====
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
            Arxora AI — современное решение, которое помогает принимать обоснованные решения
            на финансовых рынках с помощью ансамбля моделей и калибровки уверенности Octopus.
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
