# app.py — Arxora (AI) • clean Bloomberg-like skin
import os
import re
import hashlib
import random
import streamlit as st
from dotenv import load_dotenv

from core.strategy import analyze_asset

load_dotenv()

# ===================== PAGE / THEME =====================
st.set_page_config(
    page_title="Arxora — трейд-ИИ (MVP)",
    page_icon="assets/arxora_favicon_512.png",
    layout="centered",
)

# ---- one-color, strict UI ----------------------------------------------------
PRIMARY_BG   = "#0B0D0E"   # общий фон страницы (один цвет)
PANEL_BG     = "#11161B"   # фон карточек
BORDER_COLOR = "rgba(255,255,255,.08)"
GREEN_BG     = "#113628"
RED_BG       = "#3a1f20"
TEXT_MAIN    = "#FFFFFF"
TEXT_SOFT    = "rgba(255,255,255,.78)"
TEXT_MUTE    = "rgba(255,255,255,.58)"
ACCENT_ORANGE= "#FFA94D"

st.markdown(f"""
<style>
html, body, [data-testid="stApp"] {{
  background:{PRIMARY_BG} !important; color:{TEXT_MAIN} !important;
}}
.block-container {{
  padding-top: 18px; max-width: 1020px;
}}
/* убираем стандартные теневые/контрастные элементы */
.stButton>button {{
  background: #1a2733; color: {TEXT_MAIN}; border: 1px solid {BORDER_COLOR};
  border-radius: 10px; padding: 8px 14px; font-weight: 600;
}}
.stButton>button:hover {{ filter: brightness(1.08); }}
/* инпуты */
input, textarea, select, .stTextInput>div>div>input {{
  background: #121820 !important; color: {TEXT_MAIN} !important;
  border: 1px solid {BORDER_COLOR} !important; border-radius: 10px !important;
}}
/* убираем синие сообщения streamlit */
.stAlert, .stInfo, .stSuccess, .stError, .stWarning {{
  background: {PANEL_BG}; color:{TEXT_MAIN}; border:1px solid {BORDER_COLOR};
}}
/* RR — оранжевый */
.rr-line {{ color: {ACCENT_ORANGE}; font-weight: 600; }}
/* компактный заголовок */
.app-hero {{
  display:flex; gap:10px; align-items:flex-end; margin-bottom: 8px;
}}
.app-title {{ font-size: 28px; font-weight: 800; letter-spacing:.3px; }}
.app-sub   {{ font-size: 14px; color:{TEXT_MUTE}; }}
/* хедер с действием */
.header-card {{
  background:{PANEL_BG}; border:1px solid {BORDER_COLOR};
  border-radius:14px; padding:14px 16px; margin: 10px 0 8px 0;
}}
.header-title {{ font-size: 18px; font-weight: 700; }}
.header-sub   {{ font-size: 14px; color:{TEXT_SOFT}; margin-top:2px; }}
/* карточки уровней */
.card {{
  background:{PANEL_BG}; border:1px solid {BORDER_COLOR};
  border-radius:14px; padding:12px 16px; margin:6px 0;
}}
.card .t {{ font-size: 14px; color:{TEXT_SOFT}; }}
.card .v {{ font-size: 20px; font-weight: 800; margin-top: 4px; }}
.card.green {{ background:{GREEN_BG}; }}
.card.red   {{ background:{RED_BG}; }}
/* крупная цена по центру — без колец/градиентов */
.big-price {{
  font-size: 48px; font-weight: 900; text-align:center; margin: 10px 0 16px 0;
}}
/* обычный текст */
.soft  {{ color:{TEXT_SOFT}; }}
.mute  {{ color:{TEXT_MUTE}; }}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
def render_header():
    st.markdown(
        f"""
        <div class="app-hero">
          <div class="app-title">Arxora</div>
          <div class="app-sub">trade smarter.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_header()

# ===================== SETTINGS =====================
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010"))

# ===================== TEXTS =====================
CUSTOM_PHRASES = {
    "BUY": [
        "Точка входа: покупка в диапазоне {range_low}–{range_high}{unit_suffix}. По результатам AI-анализа выявлена зона поддержки на текущем горизонте."
    ],
    "SHORT": [
        "Точка входа: продажа (short) в диапазоне {range_low}–{range_high}{unit_suffix}. AI-анализ выявил зону сопротивления на текущем горизонте."
    ],
    "WAIT": [
        "Пока без позиции — жду более чёткого сигнала.",
        "Не тороплюсь: нужен более ясный сетап. Возможны новости — волатильность может измениться."
    ],
    "CONTEXT": {
        "support": ["Цена подошла к поддержке, вероятность разворота повышена. Действуем дисциплинированно, держим стоп."],
        "resistance": ["Цена у сопротивления, вероятность коррекции выше. Действуем дисциплинированно, держим стоп."],
        "neutral": ["Рынок в балансе — работаю только по подтверждённому сигналу."]
    },
    "STOPLINE": [
        "Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа. Уровень выбран по волатильности."
    ],
    "DISCLAIMER": "Пример идеи от AI, не инвестиционная рекомендация. Рынок меняется быстро; прошлые результаты не гарантируют будущих."
}

# ===================== HELPERS =====================
def _fmt(x): return f"{float(x):.2f}"

def compute_display_range(levels, widen_factor=0.25):
    entry = float(levels["entry"]); sl = float(levels["sl"])
    risk = abs(entry - sl); width = max(risk * widen_factor, 0.01)
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
    extra = ""
    if color == "green": extra = " green"
    elif color == "red": extra = " red"
    return f"""
        <div class="card{extra}">
          <div class="t">{title}</div>
          <div class="v">{value}</div>
          {f"<div class='t' style='margin-top:2px'>{sub}</div>" if sub else ""}
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

# ===================== INPUTS =====================
col1, col2 = st.columns([2,1])
with col1:
    ticker_input = st.text_input("Тикер", value="", placeholder="Примеры: AAPL · TSLA · X:BTCUSD · BTCUSDT")
    ticker = ticker_input.strip().upper()
with col2:
    horizon = st.selectbox("Горизонт",
        ["Краткосрок (1–5 дней)", "Среднесрок (1–4 недели)", "Долгосрок (1–6 месяцев)"],
        index=1)

symbol_for_engine = normalize_for_polygon(ticker)
run = st.button("Проанализировать", type="primary")

# статус строки — моно, без ярких акцентов
hz_tag = "ST" if "Кратко" in horizon else ("MID" if "Средне" in horizon else "LT")
st.markdown(f"<div class='mute'>Mode: AI · Horizon: {hz_tag}</div>", unsafe_allow_html=True)

# ===================== MAIN =====================
if run and ticker:
    try:
        out = analyze_asset(ticker=symbol_for_engine, horizon=horizon)

        last_price = float(out.get("last_price", 0.0))
        st.markdown(f"<div class='big-price'>${last_price:.2f}</div>", unsafe_allow_html=True)

        action = out["recommendation"]["action"]
        conf   = float(out["recommendation"].get("confidence", 0))
        conf_pct = f"{int(round(conf*100))}%"

        lv = dict(out["levels"])
        if action in ("BUY","SHORT"):
            t1,t2,t3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
            lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

        mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
        header_text = "WAIT"
        if action == "BUY":   header_text = f"Long • {mode_text}"
        elif action == "SHORT": header_text = f"Short • {mode_text}"

        st.markdown(
            f"""
            <div class="header-card">
                <div class="header-title">{header_text}</div>
                <div class="header-sub">{conf_pct} confidence</div>
            </div>
            """, unsafe_allow_html=True
        )

        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            with c3: st.markdown(card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out['probs']['tp1']*100))}%"), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1: st.markdown(card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out['probs']['tp2']*100))}%"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out['probs']['tp3']*100))}%"), unsafe_allow_html=True)

            rr = rr_line(lv)
            if rr: st.markdown(f"<div class='rr-line' style='margin-top:4px'>{rr}</div>", unsafe_allow_html=True)

        # план / контекст / стоп
        def render_plan_line(action, levels, ticker="", seed_extra=""):
            seed = int(hashlib.sha1(f"{ticker}{seed_extra}{levels['entry']}{levels['sl']}{action}".encode()).hexdigest(), 16) % (2**32)
            rnd = random.Random(seed)
            if action == "WAIT":
                return rnd.choice(CUSTOM_PHRASES["WAIT"])
            rng_low, rng_high = compute_display_range(levels)
            us = unit_suffix(ticker)
            tpl = CUSTOM_PHRASES[action][0]
            return tpl.format(range_low=rng_low, range_high=rng_high, unit_suffix=us)

        plan = render_plan_line(action, lv, ticker=ticker, seed_extra=horizon)
        st.markdown(f"<div class='soft' style='margin-top:8px'>{plan}</div>", unsafe_allow_html=True)

        ctx_key = "support" if action == "BUY" else ("resistance" if action == "SHORT" else "neutral")
        st.markdown(f"<div class='soft'>{CUSTOM_PHRASES['CONTEXT'][ctx_key][0]}</div>", unsafe_allow_html=True)

        if action in ("BUY","SHORT"):
            stopline = CUSTOM_PHRASES["STOPLINE"][0].format(sl=_fmt(lv["sl"]), risk_pct=compute_risk_pct(lv))
            st.markdown(f"<div class='soft' style='margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(f"<div style='margin-top:6px'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>", unsafe_allow_html=True)

        st.caption(CUSTOM_PHRASES["DISCLAIMER"])

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать».")
