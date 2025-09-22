# app.py — Arxora (AI) — agents UI with compatibility fallback
import os
import re
import hashlib
import random
import streamlit as st
from dotenv import load_dotenv

# -------- Robust imports (new API -> fallback to old) --------
try:
    from core.strategy import analyze_by_agent, Agent
    _NEW_API = True
except Exception as _IMPORT_ERR:
    _NEW_API = False
    from core.strategy import analyze_asset, analyze_asset_m7

load_dotenv()

# ===================== BRANDING =====================
st.set_page_config(
    page_title="Arxora — трейд-ИИ (MVP)",
    page_icon="assets/arxora_favicon_512.png",
    layout="centered",
)

def render_arxora_header():
    hero_path = "assets/arxora_logo_hero.png"
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)
    else:
        PURPLE = "#5B5BF7"; BLACK = "#000000"
        st.markdown(
            f"""
            <div style="border-radius:8px;overflow:hidden;
                        box-shadow:0 0 0 1px rgba(0,0,0,.06),0 12px 32px rgba(0,0,0,.18);">
              <div style="background:{PURPLE};padding:28px 16px;">
                <div style="max-width:1120px;margin:0 auto;">
                  <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
                              color:#fff;font-weight:700;letter-spacing:.4px;
                              font-size:clamp(36px,7vw,72px);line-height:1.05;">
                    Arxora
                  </div>
                </div>
              </div>
              <div style="background:{BLACK};padding:12px 16px 16px 16px;">
                <div style="max-width:1120px;margin:0 auto;">
                  <div style="color:#fff;font-size:clamp(16px,2.4vw,28px);opacity:.92;">trade smarter.</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

render_arxora_header()

# ===================== НАСТРОЙКИ UI/логики =====================
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010"))

# ===================== ТЕКСТЫ =====================
CUSTOM_PHRASES = {
    "BUY": [
        "Точка входа: покупка в диапазоне {range_low}–{range_high}{unit_suffix}. По результатам AI-анализа выявлена ключевая область спроса."
    ],
    "SHORT": [
        "Точка входа: продажа (short) в диапазоне {range_low}–{range_high}{unit_suffix}. AI-анализ выявил значимую область предложения."
    ],
    "WAIT": [
        "Пока нет ясности — лучше дождаться более чёткого сигнала.",
        "Пока не стоит спешить — дождаться подтверждения или ретеста уровня.",
        "Пока без позиции — следим за волатильностью и новостями."
    ],
    "CONTEXT": {
        "support": ["Цена у поддержки. Отложенный вход из зоны спроса, стоп за уровнем. Соблюдайте риск-менеджмент."],
        "resistance": ["Цена у сопротивления. Отложенный шорт от зоны предложения, стоп над уровнем. Соблюдайте риск-менеджмент."],
        "neutral": ["Баланс. Работаем только по подтверждённому сигналу."]
    },
    "STOPLINE": [
        "Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа. Уровень оценён по волатильности."
    ],
    "DISCLAIMER": "AI-анализ не является инвестиционной рекомендацией. Рынок волатилен; прошлые результаты не гарантируют будущие."
}

# ===================== helpers =====================
def _fmt(x): return f"{float(x):.2f}"

def compute_display_range(levels, widen_factor=0.25):
    entry = float(levels["entry"]); sl = float(levels["sl"])
    risk = abs(entry - sl)
    width = max(risk * widen_factor, 0.01)
    low, high = entry - width, entry + width
    return _fmt(min(low, high)), _fmt(max(low, high))

def compute_risk_pct(levels):
    entry = float(levels["entry"]); sl = float(levels["sl"])
    return "—" if entry == 0 else f"{abs(entry - sl)/max(1e-9,abs(entry))*100.0:.1f}"

UNIT_STYLE = {"equity":"za_akciyu","etf":"omit","crypto":"per_base","fx":"per_base","option":"per_contract"}
ETF_HINTS  = {"SPY","QQQ","IWM","DIA","EEM","EFA","XLK","XLF","XLE","XLY","XLI","XLV","XLP","XLU","VNQ","GLD","SLV"}

def detect_asset_class(ticker: str):
    t = ticker.upper().strip()
    if t.startswith("X:"): return "crypto"
    if t.startswith("C:"): return "fx"
    if t.startswith("O:"): return "option"
    if re.match(r"^[A-Z]{2,10}[-:/]?USD[TDC]?$", t): return "crypto"
    if t in ETF_HINTS: return "etf"
    return "equity"

def parse_base_symbol(ticker: str):
    t = ticker.upper().replace("X:","").replace("C:","").replace(":","").replace("/","").replace("-","")
    for q in ("USDT","USDC","USD","EUR","JPY","GBP","BTC","ETH"):
        if t.endswith(q) and len(t) > len(q): return t[:-len(q)]
    return re.split(r"[-:/]", ticker.upper())[0].replace("X:","").replace("C:","")

def unit_suffix(ticker: str) -> str:
    kind = detect_asset_class(ticker)
    style = UNIT_STYLE.get(kind, "omit")
    if style == "za_akciyu":   return " за акцию"
    if style == "per_base":     return f" за 1 {parse_base_symbol(ticker)}"
    if style == "per_contract": return " за контракт"
    return ""

def rr_line(levels):
    risk = abs(levels["entry"] - levels["sl"])
    if risk <= 1e-9: return ""
    rr1 = abs(levels["tp1"] - levels["entry"]) / risk
    rr2 = abs(levels["tp2"] - levels["entry"]) / risk
    rr3 = abs(levels["tp3"] - levels["entry"]) / risk
    return f"RR ≈ 1:{rr1:.1f} (TP1) · 1:{rr2:.1f} (TP2) · 1:{rr3:.1f} (TP3)"

def card_html(title, value, sub=None, color=None):
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
        return "Market price", "Entry (Market)"
    if action == "BUY":
        return ("Buy Stop", "Entry (Buy Stop)") if entry > last_price else ("Buy Limit", "Entry (Buy Limit)")
    else:
        return ("Sell Stop", "Entry (Sell Stop)") if entry < last_price else ("Sell Limit", "Entry (Sell Limit)")

# -------- Compatibility runner --------
def run_agent(ticker_norm: str, label: str):
    if _NEW_API:
        lbl = label.strip().lower()
        if lbl == "alphapulse":
            return analyze_by_agent(ticker_norm, Agent.ALPHAPULSE)
        if lbl == "octopus":
            return analyze_by_agent(ticker_norm, Agent.OCTOPUS)
        if lbl == "global":
            return analyze_by_agent(ticker_norm, Agent.GLOBAL)
        if lbl == "m7pro":
            return analyze_by_agent(ticker_norm, Agent.M7PRO)
        raise ValueError(f"Unknown agent label: {label}")
    else:
        # Старые API: маппинг в прежние функции
        if label == "AlphaPulse":
            return analyze_asset(ticker_norm, "Среднесрочный", strategy="W7")
        if label == "Octopus":
            return analyze_asset(ticker_norm, "Краткосрочный", strategy="W7")
        if label == "Global":
            return analyze_asset(ticker_norm, "Долгосрок", strategy="Global")
        if label == "M7pro":
            return analyze_asset_m7(ticker_norm)

# ===================== Inputs (agents) =====================

is_pro = bool(st.secrets.get("PRO_SUBSCRIBER", False)) or st.session_state.get("is_pro", False)

AGENTS = [
    {"label": "AlphaPulse", "caption": "Среднесрок (W7 • MID)", "pro": False},
    {"label": "Octopus",    "caption": "Краткосрок (W7 • ST)", "pro": True},
    {"label": "Global",     "caption": "Долгосрок (Global • LT)", "pro": False},
    {"label": "M7pro",      "caption": "Отдельный AI‑профиль", "pro": True},
]

def fmt(i: int) -> str:
    a = AGENTS[i]
    lock = " 🔒" if (a["pro"] and not is_pro) else ""
    return f'{a["label"]}{lock}'

st.subheader("AI agent")
idx = st.radio(
    "Выберите профиль",
    options=list(range(len(AGENTS))),
    index=0,
    format_func=fmt,
    captions=[a["caption"] for a in AGENTS],
    horizontal=False,
    key="agent_radio"
)

agent_rec = AGENTS[idx]

ticker_input = st.text_input(
    "Тикер",
    value="",
    placeholder="Примеры: AAPL · TSLA · X:BTCUSD · BTCUSDT",
    key="main_ticker",
)
ticker = ticker_input.strip().upper()
symbol_for_engine = normalize_for_polygon(ticker)

run = st.button("Проанализировать", type="primary", key="main_analyze")

st.write(f"Mode: AI · Agent: {agent_rec['label']}")

# ===================== Main =====================
if run and ticker:
    if agent_rec["pro"] and not is_pro:
        st.info("Этот агент доступен по подписке. Оформите PRO, чтобы разблокировать Octopus и M7pro.", icon="🔒")
        st.stop()
    try:
        out = run_agent(symbol_for_engine, agent_rec["label"])

        last_price = float(out.get("last_price", 0.0))
        st.markdown(
            f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>${last_price:.2f}</div>",
            unsafe_allow_html=True,
        )

        action = out["recommendation"]["action"]
        conf   = float(out["recommendation"].get("confidence", 0))
        conf_pct = f"{int(round(conf*100))}%"

        lv = dict(out["levels"])
        if action in ("BUY","SHORT"):
            t1,t2,t3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
            lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

        mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
        header_text = "WAIT"
        if action == "BUY":
            header_text = f"Long • {mode_text}"
        elif action == "SHORT":
            header_text = f"Short • {mode_text}"

        st.markdown(
            f"""
            <div style="background:#c57b0a; padding:14px 16px; border-radius:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:10px;">
                <div style="font-size:1.15rem; font-weight:700;">{header_text}</div>
                <div style="opacity:0.75; font-size:0.95rem; margin-top:2px;">{conf_pct} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            with c3: st.markdown(card_html("TP 1", f"{lv['tp1']:.2f}",
                                           sub=f"Probability {int(round(out['probs']['tp1']*100))}%"),
                                  unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1: st.markdown(card_html("TP 2", f"{lv['tp2']:.2f}",
                                           sub=f"Probability {int(round(out['probs']['tp2']*100))}%"),
                                  unsafe_allow_html=True)
            with c2: st.markdown(card_html("TP 3", f"{lv['tp3']:.2f}",
                                           sub=f"Probability {int(round(out['probs']['tp3']*100))}%"),
                                  unsafe_allow_html=True)

            rr = rr_line(lv)
            if rr:
                st.markdown(
                    f"<div style='margin-top:4px; color:#FFA94D; font-weight:600;'>{rr}</div>",
                    unsafe_allow_html=True,
                )

        def render_plan_line(action, levels, ticker="", seed_extra=""):
            seed = int(hashlib.sha1(f"{ticker}{seed_extra}{levels['entry']}{levels['sl']}{action}".encode()).hexdigest(), 16) % (2**32)
            rnd = random.Random(seed)
            if action == "WAIT":
                return rnd.choice(CUSTOM_PHRASES["WAIT"])
            rng_low, rng_high = compute_display_range(levels)
            us = unit_suffix(ticker)
            tpl = CUSTOM_PHRASES[action][0]
            return tpl.format(range_low=rng_low, range_high=rng_high, unit_suffix=us)

        plan = render_plan_line(action, lv, ticker=ticker, seed_extra=agent_rec["label"])
        st.markdown(f"<div style='margin-top:8px'>{plan}</div>", unsafe_allow_html=True)

        ctx_key = "support" if action == "BUY" else ("resistance" if action == "SHORT" else "neutral")
        st.markdown(f"<div style='opacity:0.9'>{CUSTOM_PHRASES['CONTEXT'][ctx_key][0]}</div>", unsafe_allow_html=True)

        if action in ("BUY","SHORT"):
            stopline = CUSTOM_PHRASES["STOPLINE"][0].format(sl=_fmt(lv["sl"]), risk_pct=compute_risk_pct(lv))
            st.markdown(f"<div style='opacity:0.9; margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(
                f"<div style='margin-top:6px;'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>",
                unsafe_allow_html=True,
            )

        st.caption(CUSTOM_PHRASES["DISCLAIMER"])

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать».")

# ===================== НИЖНИЙ КОЛОНТИТУЛ =====================
st.markdown("---")

st.markdown("""
<style>
    .stButton > button {
        font-weight: 600;
    }
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
            Arxora AI помогает принимать решения на рынках с помощью ИИ и ML. 
            Платформа ускоряет анализ активов, автоматизирует входы и управление риском. 
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
            Анализ основных криптовалют теми же алгоритмическими подходами, с учётом круглосуточного рынка и высокой волатильности.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

