# app.py — Arxora (AI) — Bloomberg-ish UI
import os
import re
import hashlib
import random
import streamlit as st
from dotenv import load_dotenv

# Базовый движок сигналов (AI + правила)
from core.strategy import analyze_asset

load_dotenv()

# ===================== BRANDING / THEME =====================
st.set_page_config(
    page_title="Arxora — трейд-ИИ (MVP)",
    page_icon="assets/arxora_favicon_512.png",
    layout="centered",
)

# Глобальные стили (тёмный терминал, белый текст, аккуратные карточки)
st.markdown("""
<style>
/* базовый тёмный фон */
html, body, .block-container { background: #0B0D0E !important; }
.block-container { padding-top: 14px; max-width: 980px; }

/* белый текст по умолчанию */
body, p, div, span, li, label, .stMarkdown, .stText, .stRadio, .stSelectbox, .stCaption { color: #E8EAED !important; }

/* инпуты/селекты темнее */
.stTextInput>div>div>input, .stSelectbox>div>div>div>div { color:#E8EAED !important; background:#111416 !important; }
.stTextInput>div>div, .stSelectbox>div>div { border:1px solid rgba(255,255,255,0.08) !important; }

/* кнопка primary */
button[kind="primary"]{
  background:#1a73e8 !important; color:#fff !important; border:0 !important;
  border-radius:10px !important; padding:0.6rem 1.0rem !important;
}

/* карточки */
.arx-card{
  background:#12161A; border:1px solid rgba(255,255,255,0.08);
  border-radius:14px; padding:12px 14px; margin:6px 0;
}
.arx-card .title{ font-size:0.9rem; opacity:0.85; }
.arx-card .value{ font-size:1.35rem; font-weight:700; margin-top:4px; }
.arx-card .sub  { font-size:0.80rem; opacity:0.70; margin-top:2px; }

/* цветовые варианты карточек */
.arx-card.green{ background:#0F2219; }
.arx-card.red  { background:#261314; }

/* price (моноширинный крупный) */
.arx-price{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;
}

/* шапка рекомендации */
.arx-ribbon{
  border:1px solid rgba(255,255,255,0.08);
  border-radius:16px; padding:14px 16px; margin-bottom:12px;
}
.arx-ribbon.wait  { background:#15171A; }
.arx-ribbon.long  { background:#0E2118; }
.arx-ribbon.short { background:#221212; }
.arx-ribbon .h   { font-size:1.12rem; font-weight:700; }
.arx-ribbon .s   { opacity:.75; font-size:0.95rem; margin-top:2px; }

/* RR — фирменный оранжевый */
.arx-rr{
  margin-top:4px; color:#FFB300; 
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* мелкие подсказки */
.small-dim { opacity:.88; }
</style>
""", unsafe_allow_html=True)

def render_arxora_header():
    hero_path = "assets/arxora_logo_hero.png"
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)
    else:
        PURPLE = "#5B5BF7"; BLACK = "#0B0D0E"
        st.markdown(
            f"""
            <div style="border-radius:8px;overflow:hidden;
                        box-shadow:0 0 0 1px rgba(0,0,0,.06),0 12px 32px rgba(0,0,0,.18);">
              <div style="background:{PURPLE};padding:22px 16px;">
                <div style="max-width:1120px;margin:0 auto;">
                  <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
                              color:#fff;font-weight:700;letter-spacing:.4px;
                              font-size:clamp(32px,6vw,64px);line-height:1.05;">
                    Arxora
                  </div>
                </div>
              </div>
              <div style="background:{BLACK};padding:10px 16px 14px 16px;">
                <div style="max-width:1120px;margin:0 auto;">
                  <div style="color:#fff;font-size:clamp(15px,2.2vw,24px);opacity:.92;">trade smarter.</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True,
        )

render_arxora_header()

# ===================== НАСТРОЙКИ UI/логики =====================
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))  # ≈0.15%
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010"))   # 0.10% от entry

# ===================== ТЕКСТЫ =====================
CUSTOM_PHRASES = {
    "BUY": [
        "Точка входа: покупка в диапазоне {range_low}–{range_high}{unit_suffix}. По результатам AI-анализа выявлена зона поддержки на текущем горизонте."
    ],
    "SHORT": [
        "Точка входа: продажа (short) в диапазоне {range_low}–{range_high}{unit_suffix}. AI-анализ выделяет зону сопротивления на текущем горизонте."
    ],
    "WAIT": [
        "Пока не вижу для себя ясной картины и не тороплюсь с решениями.",
        "Пока не стоит спешить — лучше дождаться более ясной картины. Вероятные новости могут изменить динамику и волатильность.",
        "Пока без позиции — жду более чёткого сигнала. Новостной фон способен сдвинуть рынок и усилить волатильность.",
        "Пока нет ясности по текущему горизонту — подожду подтверждения."
    ],
    "CONTEXT": {
        "support": ["Цена у поддержки — вероятность разворота повышена. Длинная позиция уместна при соблюдении дисциплины риска; закрепление ниже зоны — сигнал пересмотра сценария."],
        "resistance": ["Цена у сопротивления — вероятность коррекции повышена. Короткая позиция уместна при строгом контроле риска; закрепление выше зоны — сигнал пересмотра сценария."],
        "neutral": ["Рынок в балансе — действую только по подтверждённому сигналу."]
    },
    "STOPLINE": [
        "Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа. Уровень выбран с учётом волатильности для защиты капитала."
    ],
    "DISCLAIMER": "Материал носит информационный характер и не является инвестрекомендацией. Рынок подвержен рискам; прошлые результаты не гарантируют будущих."
}

# ===================== helper'ы =====================
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
    cls = "arx-card"
    if color == "green": cls += " green"
    if color == "red":   cls += " red"
    return f"""
        <div class="{cls}">
            <div class="title">{title}</div>
            <div class="value">{value}</div>
            {f"<div class='sub'>{sub}</div>" if sub else ""}
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

# Санити-правки целей (TP по направлению + порядок + минимальный шаг)
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

# Режим входа в шапке:
# BUY:  entry > price -> Buy Stop; entry < price -> Buy Limit; иначе Market
# SHORT: entry < price -> Sell Stop; entry > price -> Sell Limit; иначе Market
def entry_mode_labels(action: str, entry: float, last_price: float, eps: float):
    if action not in ("BUY", "SHORT"):
        return "WAIT", "Entry"
    if abs(entry - last_price) <= eps * max(1.0, abs(last_price)):
        return "Market price", "Entry (Market)"
    if action == "BUY":
        return ("Buy Stop", "Entry (Buy Stop)") if entry > last_price else ("Buy Limit", "Entry (Buy Limit)")
    else:
        return ("Sell Stop", "Entry (Sell Stop)") if entry < last_price else ("Sell Limit", "Entry (Sell Limit)")

# ===================== Inputs =====================
col1, col2 = st.columns([2,1])
with col1:
    ticker_input = st.text_input(
        "Тикер",
        value="",
        placeholder="Примеры: AAPL · TSLA · X:BTCUSD · BTCUSDT",
        key="main_ticker",
    )
    ticker = ticker_input.strip().upper()
with col2:
    horizon = st.selectbox(
        "Горизонт",
        ["Краткосрок (1–5 дней)", "Среднесрок (1–4 недели)", "Долгосрок (1–6 месяцев)"],
        index=1,
        key="main_horizon",
    )

symbol_for_engine = normalize_for_polygon(ticker)
run = st.button("Проанализировать", type="primary", key="main_analyze")

# Статус режима (AI/AI pseudo)
AI_PSEUDO = str(os.getenv("ARXORA_AI_PSEUDO", "0")).strip() in ("1", "true", "True", "yes")
hz_tag = "ST" if "Кратко" in horizon else ("MID" if "Средне" in horizon else "LT")
st.caption(f"Mode: {'AI (pseudo)' if AI_PSEUDO else 'AI'} · Horizon: {hz_tag}")

# ===================== Main =====================
if run and ticker:
    try:
        out = analyze_asset(ticker=symbol_for_engine, horizon=horizon)

        last_price = float(out.get("last_price", 0.0))
        st.markdown(f"<div class='arx-price'>${last_price:.2f}</div>", unsafe_allow_html=True)

        action = out["recommendation"]["action"]
        conf   = float(out["recommendation"].get("confidence", 0))
        conf_pct = f"{int(round(conf*100))}%"

        lv = dict(out["levels"])
        if action in ("BUY","SHORT"):
            t1,t2,t3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
            lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

        # шапка — цвет по направлению
        mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
        if action == "BUY":
            ribbon_cls, header_text = "arx-ribbon long",  f"Long • {mode_text}"
        elif action == "SHORT":
            ribbon_cls, header_text = "arx-ribbon short", f"Short • {mode_text}"
        else:
            ribbon_cls, header_text = "arx-ribbon wait",  "WAIT"

        st.markdown(
            f"""
            <div class="{ribbon_cls}">
                <div class="h">{header_text}</div>
                <div class="s">{conf_pct} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # карточки уровней
        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            with c3: st.markdown(
                card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out['probs']['tp1']*100))}%"),
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)
            with c1: st.markdown(
                card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out['probs']['tp2']*100))}%"),
                unsafe_allow_html=True
            )
            with c2: st.markdown(
                card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out['probs']['tp3']*100))}%"),
                unsafe_allow_html=True
            )

            rr = rr_line(lv)
            if rr: st.markdown(f"<div class='arx-rr'>{rr}</div>", unsafe_allow_html=True)

        # План / контекст / стоп-линия
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
        st.markdown(f"<div style='margin-top:8px'>{plan}</div>", unsafe_allow_html=True)

        ctx_key = "support" if action == "BUY" else ("resistance" if action == "SHORT" else "neutral")
        st.markdown(f"<div class='small-dim'>{CUSTOM_PHRASES['CONTEXT'][ctx_key][0]}</div>", unsafe_allow_html=True)

        if action in ("BUY","SHORT"):
            stopline = CUSTOM_PHRASES["STOPLINE"][0].format(sl=_fmt(lv["sl"]), risk_pct=compute_risk_pct(lv))
            st.markdown(f"<div class='small-dim' style='margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(
                f"<div style='margin-top:6px'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>",
                unsafe_allow_html=True,
            )

        st.caption(CUSTOM_PHRASES["DISCLAIMER"])

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать».")
