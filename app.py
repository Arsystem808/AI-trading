# app.py — Arxora (AI) — Premium Glass UI (wow-версия)
import os
import re
import hashlib
import random
import streamlit as st
from dotenv import load_dotenv

# Движок сигналов
from core.strategy import analyze_asset

load_dotenv()

# ===================== PAGE / THEME =====================
st.set_page_config(
    page_title="Arxora — трейд-ИИ (MVP)",
    page_icon="assets/arxora_favicon_512.png",
    layout="centered",
)

# Глобальный стиль (стекло, градиенты, неон)
st.markdown("""
<style>
/* ---- Фон: глубокий тёмный с мягким градиентом и «шумом» ---- */
html, body, .block-container { background: #090B0E !important; }
.block-container { padding-top: 14px; max-width: 1024px; }
body::before{
  content:""; position: fixed; inset: -20%;
  background: radial-gradient(1200px 700px at 10% -10%, rgba(50,90,255,.22), transparent 60%),
              radial-gradient(900px 700px at 110% 0%, rgba(130,40,180,.18), transparent 60%),
              radial-gradient(800px 700px at 30% 120%, rgba(20,220,160,.10), transparent 60%);
  filter: blur(40px); pointer-events:none; z-index: -1;
}
body::after{
  content:""; position: fixed; inset:0;
  background-image: url('data:image/svg+xml;utf8,\
  <svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" viewBox="0 0 32 32">\
    <circle cx="1" cy="1" r="1" fill="rgba(255,255,255,0.03)"/>\
  </svg>');
  opacity:.6; pointer-events:none; z-index:-1;
}

/* ---- Белый текст повсюду ---- */
*, .stMarkdown, .stText, .stCaption, .stSelectbox, .stRadio, label { color:#EDF1F5 !important; }

/* ---- Инпуты / селекты ---- */
.stTextInput>div>div>input, .stSelectbox div[data-baseweb="select"] input {
  color:#EDF1F5 !important; background:rgba(255,255,255,0.03) !important;
}
.stTextInput>div>div, .stSelectbox>div>div {
  border:1px solid rgba(255,255,255,0.12) !important; border-radius:12px !important;
  backdrop-filter: blur(8px) saturate(140%);
}

/* ---- Кнопки ---- */
button[kind="primary"]{
  background: linear-gradient(90deg, #3B82F6, #8B5CF6, #22C55E);
  color:#fff !important; border:0; border-radius:12px !important;
  padding:.65rem 1.05rem !important; transition: transform .06s ease;
}
button[kind="primary"]:hover{ transform: translateY(-1px); filter: brightness(1.06); }

/* ---- Стеклянные панели ---- */
.glass {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 16px; padding: 14px 16px;
  box-shadow:
    0 12px 28px rgba(0,0,0,.38),
    inset 0 1px 0 rgba(255,255,255,.05);
  backdrop-filter: blur(10px) saturate(160%);
}

/* ---- Хиро-карточка (логотип) ---- */
.hero {
  overflow:hidden;
  background: linear-gradient(135deg, rgba(91,91,247,.25), rgba(9,11,14,.0));
  border:1px solid rgba(91,91,247,.35);
}

/* ---- Цена (моно + неоновое кольцо) ---- */
.arx-price-wrap{ display:flex; justify-content:center; margin: 8px 0 16px 0; }
.arx-price-ring{
  --c1: #4CC9F0; --c2:#F72585;
  width: 220px; height: 220px; border-radius: 999px; padding: 14px;
  background: conic-gradient(from 210deg, var(--c1), var(--c2), var(--c1));
  filter: drop-shadow(0 10px 25px rgba(0,0,0,.5));
  animation: spinHue 9s linear infinite;
}
@keyframes spinHue { 0%{filter:hue-rotate(0deg) drop-shadow(0 10px 25px rgba(0,0,0,.5));}
                     100%{filter:hue-rotate(360deg) drop-shadow(0 10px 25px rgba(0,0,0,.5));} }
.arx-price-inner{
  background: rgba(9,11,14,.9);
  border:1px solid rgba(255,255,255,.10);
  backdrop-filter: blur(12px) saturate(140%);
  width:100%; height:100%; border-radius: inherit;
  display:flex; align-items:center; justify-content:center;
}
.arx-price{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 2.6rem; font-weight:800; letter-spacing: .5px; color:#EAF2FF;
}

/* ---- Риббон-статус (Long/Short/Wait) ---- */
.ribbon { border-radius:16px; padding:14px 16px; }
.ribbon.long  { background: linear-gradient(180deg, rgba(16,56,38,.85), rgba(10,31,23,.75)); border:1px solid rgba(34,197,94,.35); }
.ribbon.short { background: linear-gradient(180deg, rgba(56,16,24,.85), rgba(31,10,14,.75)); border:1px solid rgba(244,63,94,.35); }
.ribbon.wait  { background: linear-gradient(180deg, rgba(26,31,36,.85), rgba(18,22,26,.75)); border:1px solid rgba(255,255,255,.14); }
.ribbon .h { font-size:1.08rem; font-weight:800; letter-spacing:.2px; }
.ribbon .s { opacity:.82; font-size:.96rem; margin-top:2px; }

/* ---- Пилюли (чипы) ---- */
.pill{
  display:inline-block; padding:.28rem .55rem; border-radius:999px;
  background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.15);
  margin-right:.35rem; font-size:.78rem; letter-spacing:.2px;
}

/* ---- Карточки уровней (стекло + цвет) ---- */
.card {
  border-radius: 14px; padding: 12px 14px; margin: 6px 0;
  background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.14);
  backdrop-filter: blur(8px) saturate(140%);
}
.card.green { background: linear-gradient(180deg, rgba(21,50,39,.7), rgba(16,36,28,.6)); border-color: rgba(34,197,94,.28); }
.card.red   { background: linear-gradient(180deg, rgba(52,20,22,.7), rgba(37,14,16,.6)); border-color: rgba(244,63,94,.28); }
.card .t { font-size:.9rem; opacity:.85; }
.card .v { font-size:1.35rem; font-weight:800; margin-top:2px; color:#FAFAFC; text-shadow: 0 0 10px rgba(255,255,255,.04); }
.card .s { font-size:.80rem; opacity:.75; margin-top:2px; }

/* ---- RR — оранжевый, моно ---- */
.arx-rr{
  margin-top:6px; color:#FFB300; 
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  background: rgba(255,179,0,.08);
  border: 1px solid rgba(255,179,0,.25);
  border-radius: 10px; padding: 6px 10px; display:inline-block;
}

/* ---- Мелкие тексты ---- */
.dim { opacity: .9; }
.caption { opacity:.75; }
</style>
""", unsafe_allow_html=True)

# ===================== Header =====================
def render_arxora_header():
    hero_path = "assets/arxora_logo_hero.png"
    if os.path.exists(hero_path):
        st.markdown('<div class="glass hero" style="padding:18px 16px; margin-bottom:12px;">', unsafe_allow_html=True)
        st.image(hero_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="glass hero" style="padding:18px 16px; margin-bottom:12px;">
              <div style="font-weight:800; font-size: clamp(32px, 6vw, 54px); letter-spacing:.4px;">Arxora</div>
              <div style="opacity:.92; font-size:clamp(14px,2.1vw,22px);">trade smarter.</div>
            </div>
            """, unsafe_allow_html=True
        )

render_arxora_header()

# ===================== UI / LOGIC SETTINGS =====================
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))  # ≈0.15%
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010"))   # 0.10% от entry

# ===================== TEXTS =====================
CUSTOM_PHRASES = {
    "BUY": [
        "Точка входа: покупка в диапазоне {range_low}–{range_high}{unit_suffix}. По результатам AI-анализа выделена зона поддержки на текущем горизонте."
    ],
    "SHORT": [
        "Точка входа: продажа (short) в диапазоне {range_low}–{range_high}{unit_suffix}. AI-анализ указывает на зону сопротивления на текущем горизонте."
    ],
    "WAIT": [
        "Пока не вижу для себя ясной картины и не тороплюсь с решениями.",
        "Пока не стоит спешить — лучше дождаться более ясной картины. Вероятные новости могут изменить волатильность.",
        "Пока без позиции — жду более чёткого сигнала. Новостной фон может сдвинуть рынок и усилить импульс.",
        "Пока нет ясности по текущему горизонту — подожду подтверждения."
    ],
    "CONTEXT": {
        "support": ["Цена у поддержки — вероятность разворота повышена. Длинная позиция уместна при соблюдении дисциплины риска; закрепление ниже зоны — повод пересмотреть сценарий."],
        "resistance": ["Цена у сопротивления — вероятность коррекции повышена. Короткая позиция уместна при строгом контроле риска; закрепление выше зоны — сигнал к пересмотру сценария."],
        "neutral": ["Рынок в балансе — действую только по подтверждённому сигналу."]
    },
    "STOPLINE": [
        "Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа. Уровень выбран с учётом волатильности для защиты капитала."
    ],
    "DISCLAIMER": "Материал носит информационный характер и не является инвестрекомендацией. Рынок подвержен рискам; прошлые результаты не гарантируют будущих."
}

# ===================== HELPERS =====================
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
    cls = "card"
    if color == "green": cls += " green"
    if color == "red":   cls += " red"
    return f"""
      <div class="{cls}">
        <div class="t">{title}</div>
        <div class="v">{value}</div>
        {f'<div class="s">{sub}</div>' if sub else ""}
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

# Санити-правки TP (по направлению + упорядочены + минимальный шаг)
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

# Режим входа для шапки
# BUY:   entry > price -> Buy Stop;  entry < price -> Buy Limit;  иначе Market
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

# Режим (AI/pseudo)
AI_PSEUDO = str(os.getenv("ARXORA_AI_PSEUDO", "0")).strip() in ("1", "true", "True", "yes")
hz_tag = "ST" if "Кратко" in horizon else ("MID" if "Средне" in horizon else "LT")
st.markdown(
    f'<div class="glass" style="display:flex;gap:.5rem;align-items:center;justify-content:space-between;margin:6px 0 10px 0;">'
    f'<div><span class="pill">Mode: {"AI (pseudo)" if AI_PSEUDO else "AI"}</span>'
    f'<span class="pill">Horizon: {hz_tag}</span></div>'
    f'<div class="caption">ENTRY_MARKET_EPS={ENTRY_MARKET_EPS:.4f} · MIN_TP_STEP_PCT={MIN_TP_STEP_PCT:.4f}</div>'
    f'</div>', unsafe_allow_html=True
)

# ===================== Main =====================
if run and ticker:
    try:
        out = analyze_asset(ticker=symbol_for_engine, horizon=horizon)

        last_price = float(out.get("last_price", 0.0))
        st.markdown(
            f"""
            <div class="arx-price-wrap">
              <div class="arx-price-ring">
                <div class="arx-price-inner">
                  <div class="arx-price">${last_price:.2f}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        action = out["recommendation"]["action"]
        conf   = float(out["recommendation"].get("confidence", 0))
        conf_pct = f"{int(round(conf*100))}%"

        lv = dict(out["levels"])
        if action in ("BUY","SHORT"):
            t1,t2,t3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
            lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

        # Риббон
        mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
        if action == "BUY":
            rib_cls, head_text = "ribbon long",  f"Long • {mode_text}"
        elif action == "SHORT":
            rib_cls, head_text = "ribbon short", f"Short • {mode_text}"
        else:
            rib_cls, head_text = "ribbon wait",  "WAIT"

        st.markdown(
            f"""
            <div class="{rib_cls}">
              <div class="h">{head_text}</div>
              <div class="s">{conf_pct} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Карточки уровней
        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2:
                st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            with c3:
                st.markdown(
                    card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out['probs']['tp1']*100))}%"),
                    unsafe_allow_html=True
                )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out['probs']['tp2']*100))}%"),
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out['probs']['tp3']*100))}%"),
                    unsafe_allow_html=True
                )

            rr = rr_line(lv)
            if rr:
                st.markdown(f"<div class='arx-rr'>{rr}</div>", unsafe_allow_html=True)

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

        # стеклянный контейнер под текстовые блоки
        st.markdown('<div class="glass" style="margin-top:10px">', unsafe_allow_html=True)

        plan = render_plan_line(action, lv, ticker=ticker, seed_extra=horizon)
        st.markdown(f"<div>{plan}</div>", unsafe_allow_html=True)

        ctx_key = "support" if action == "BUY" else ("resistance" if action == "SHORT" else "neutral")
        st.markdown(f"<div class='dim' style='margin-top:6px'>{CUSTOM_PHRASES['CONTEXT'][ctx_key][0]}</div>", unsafe_allow_html=True)

        if action in ("BUY","SHORT"):
            stopline = CUSTOM_PHRASES["STOPLINE"][0].format(sl=_fmt(lv["sl"]), risk_pct=compute_risk_pct(lv))
            st.markdown(f"<div class='dim' style='margin-top:6px'>{stopline}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(f"<div style='margin-top:8px'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(CUSTOM_PHRASES["DISCLAIMER"])

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать».")
