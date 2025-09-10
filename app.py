# app.py — Arxora (AI)
import os
import re
import hashlib
import random
import streamlit as st
from dotenv import load_dotenv

# Базовый движок сигналов (AI + правила)
from core.strategy import analyze_asset

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
        # Строгий fallback-хедер без ярких градиентов
        st.markdown(
            """
            <div style="border-radius:10px; padding:18px 16px;
                        background:#0b0d0e; border:1px solid rgba(255,255,255,.06);
                        box-shadow:0 6px 26px rgba(0,0,0,.35);">
              <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
                          color:#ffffff; font-weight:800; letter-spacing:.4px;
                          font-size:clamp(30px,6vw,58px); line-height:1.05;">
                Arxora
              </div>
              <div style="color:#aab6c6; font-size:clamp(14px,2.1vw,22px); opacity:.92;">trade smarter.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

render_arxora_header()

# ===================== НАСТРОЙКИ UI/логики =====================
# Эпсилон для распознавания "рыночного" входа (в долях цены): 0.0015 ≈ 0.15%
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))
# Минимальный шаг для выравнивания целей (в долях от entry)
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010"))

# ===================== ТЕКСТЫ =====================
CUSTOM_PHRASES = {
    "BUY": [
        "Точка входа: покупка в диапазоне {range_low}–{range_high}{unit_suffix}. По результатам AI-анализа выявлена зона поддержки в рамках текущего временного горизонта."
    ],
    "SHORT": [
        "Точка входа: продажа (short) в диапазоне {range_low}–{range_high}{unit_suffix}. AI-анализ выявил зону сопротивления на текущем временном горизонте."
    ],
    "WAIT": [
        "Пока не вижу для себя ясной картины и не тороплюсь с решениями.",
        "Пока не стоит спешить — лучше дождаться более ясной картины. Вероятно, новости могут спровоцировать изменения на рынке.",
        "Пока без позиции — жду более чёткого сигнала. Новостной фон может сдвинуть рынок и повысить волатильность.",
        "Пока нет ясности с картиной на текущем горизонте, и я не спешу с решениями."
    ],
    "CONTEXT": {
        "support": ["Цена подошла к поддержке, и вероятность разворота здесь повышена. Оптимальный сценарий — аккуратный лонг с контролем риска. Закрепление ниже зоны будет сигналом к пересмотру сценария."],
        "resistance": ["Цена подошла к сопротивлению, и вероятность коррекции здесь повышена. Оптимальный сценарий — работа от шорта со стопом. Пробой и удержание выше — повод пересмотреть сценарий."],
        "neutral": ["Рынок пока в балансе — действую только по подтверждённому сигналу."]
    },
    "STOPLINE": [
        "Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа. Уровень определён алгоритмами анализа волатильности как критический для защиты капитала."
    ],
    "DISCLAIMER": "Данная информация является примером того, как AI может генерировать инвестиционные идеи и не является прямой инвестиционной рекомендацией. Рыночная ситуация может быстро меняться; прошлые результаты не гарантируют будущих. Торговля на финансовых рынках сопряжена с высоким риском."
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

# ----- Палитра плиток/шапки -----
CLR_BLUE_1   = "#0ea5ff"   # cyan
CLR_BLUE_2   = "#2563eb"   # blue
CLR_ENTRY_BG = "#123e2a"   # зелёная плитка entry
CLR_ENTRY_BR = "rgba(34,197,94,.45)"    # зелёная окантовка
CLR_SL_BG    = "#3b1f20"   # красная плитка stop
CLR_SL_BR    = "rgba(239,68,68,.45)"    # красная окантовка
CLR_TP_BG    = "#1a2027"   # тёмная плитка TP
CLR_TP_BR    = "rgba(255,255,255,.06)"
CLR_NEU_TXT  = "rgba(255,255,255,.85)"

# ----- карточки -----
def card_html(title, value, sub=None, variant="neutral"):
    if variant == "entry":
        bg, br = CLR_ENTRY_BG, CLR_ENTRY_BR
    elif variant == "sl":
        bg, br = CLR_SL_BG, CLR_SL_BR
    elif variant == "tp":
        bg, br = CLR_TP_BG, CLR_TP_BR
    else:
        bg, br = "#141a20", "rgba(255,255,255,.06)"

    return f"""
        <div style="
            background:{bg};
            padding:12px 16px;
            border-radius:14px;
            border:1px solid {br};
            box-shadow:0 6px 22px rgba(0,0,0,.28);
            margin:6px 0;">
            <div style="font-size:0.9rem; color:{CLR_NEU_TXT};">{title}</div>
            <div style="font-size:1.45rem; font-weight:700; margin-top:4px; color:#fff;">
                {value}
            </div>
            {f"<div style='font-size:0.8rem; opacity:.75; margin-top:2px;'>{sub}</div>" if sub else ""}
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

# Санити-правки целей (TP по направлению + по порядку)
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

# Режим входа для шапки/карточки Entry:
# BUY:  entry > price -> Buy Stop; entry < price -> Buy Limit; иначе Market
# SHORT: entry < price -> Sell Stop; entry > price -> Sell Limit; иначе Market
def entry_mode_labels(action: str, entry: float, last_price: float, eps: float):
    if action not in ("BUY", "SHORT"):
        return "WAIT", "Entry"
    # “Практически рынок”
    if abs(entry - last_price) <= eps * max(1.0, abs(last_price)):
        return "Market price", "Entry (Market)"
    if action == "BUY":
        return ("Buy Stop", "Entry (Buy Stop)") if entry > last_price else ("Buy Limit", "Entry (Buy Limit)")
    else:  # SHORT
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
st.write(f"Mode: {'AI (pseudo)' if AI_PSEUDO else 'AI'} · Horizon: {hz_tag}")

# ===================== Main =====================
if run and ticker:
    try:
        out = analyze_asset(ticker=symbol_for_engine, horizon=horizon)

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

        # --- шапка: Long/Short + Buy/Sell Stop/Limit/Market (сине-голубая)
        mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
        header_text = "WAIT"
        if action == "BUY":
            header_text = f"Long • {mode_text}"
        elif action == "SHORT":
            header_text = f"Short • {mode_text}"

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg, {CLR_BLUE_1}26 0%, {CLR_BLUE_2}26 100%);
                border:1px solid rgba(14,165,233,.35);
                padding:14px 16px;
                border-radius:16px;
                box-shadow:0 10px 30px rgba(0,0,0,.35);
                margin-bottom:10px;">
                <div style="font-size:1.15rem; font-weight:700; color:#e8f2ff;">{header_text}</div>
                <div style="opacity:.80; font-size:0.95rem; margin-top:2px; color:#cfe6ff;">
                    {conf_pct} confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- карточки уровней (Entry зелёный, Stop красный, TP тёмные)
        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", variant="entry"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", variant="sl"), unsafe_allow_html=True)
            with c3: st.markdown(
                card_html("TP 1", f"{lv['tp1']:.2f}",
                          sub=f"Probability {int(round(out['probs']['tp1']*100))}%", variant="tp"),
                unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1: st.markdown(
                card_html("TP 2", f"{lv['tp2']:.2f}",
                          sub=f"Probability {int(round(out['probs']['tp2']*100))}%", variant="tp"),
                unsafe_allow_html=True)
            with c2: st.markdown(
                card_html("TP 3", f"{lv['tp3']:.2f}",
                          sub=f"Probability {int(round(out['probs']['tp3']*100))}%", variant="tp"),
                unsafe_allow_html=True)

            rr = rr_line(lv)
            if rr:
                st.markdown(
                    f"<div style='margin-top:4px; color:#ffb454; font-weight:600;'>{rr}</div>",
                    unsafe_allow_html=True,
                )

        # --- план/контекст/стоп-линия
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
