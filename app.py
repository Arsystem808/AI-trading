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

# ===================== BRANDING / THEME =====================
st.set_page_config(
    page_title="Arxora — трейд-ИИ (MVP)",
    page_icon="assets/arxora_favicon_512.png",
    layout="centered",
)

# Глобальные стили (единый фон, белый текст, строгая подача)
st.markdown(
    """
    <style>
      :root{
        --bg:#0b1117;
        --panel:#0f1621;
        --card:#111a24;
        --text:#e6edf3;
        --muted:#9aa4b2;
        --green:#22c55e;
        --red:#ef4444;
        --blue:#3b82f6;
        --blue2:#0ea5e9;
        --orange:#f59e0b;
        --border:rgba(255,255,255,.06);
      }
      html, body, [data-testid="stApp"]{
        background:var(--bg)!important;
        color:var(--text)!important;
      }
      [data-testid="stHeader"]{background:transparent!important}
      .stButton>button{
        background:linear-gradient(90deg, var(--blue2), var(--blue));
        border:0;
        color:#fff;
        font-weight:700;
        border-radius:12px;
      }
      .arxora-pill{
        background:linear-gradient(90deg, rgba(14,165,233,.28), rgba(59,130,246,.28));
        border:1px solid var(--border);
        padding:14px 16px;
        border-radius:14px;
      }
      .arxora-h-num{
        font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;
      }
      .arxora-card{
        background:var(--card); padding:12px 16px;
        border-radius:14px; border:1px solid var(--border); margin:6px 0;
      }
      .arxora-card .t{font-size:.92rem; opacity:.9}
      .arxora-card .v{font-size:1.4rem; font-weight:700; margin-top:4px}
      .arxora-card .s{font-size:.82rem; opacity:.72; margin-top:2px}
      .entry{box-shadow:0 0 0 1px rgba(34,197,94,.15) inset}
      .entry .v{color:var(--green)}
      .sl{box-shadow:0 0 0 1px rgba(239,68,68,.18) inset}
      .sl .v{color:var(--red)}
      .tp{box-shadow:0 0 0 1px rgba(59,130,246,.14) inset}
      .muted{opacity:.9}
      .rr-line{margin-top:4px; color:var(--orange); font-weight:600}
      .note, .ctx, .disc{opacity:.9}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_arxora_header():
    hero_path = "assets/arxora_logo_hero.png"
    if os.getenv("ARXORA_SHOW_HERO", "0").strip() in ("1", "true", "True", "yes") and os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)

# По умолчанию — без яркого баннера
render_arxora_header()

# ===================== НАСТРОЙКИ UI/логики =====================
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))  # ~0.15%
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
        "support": ["Цена подошла к поддержке, и вероятность разворота здесь повышена. Оптимальный вариант — длинная позиция с расчётом на рост. Закрепление ниже зоны — сигнал пересмотра сценария. Дисциплина и стоп-лосс обязательны."],
        "resistance": ["Цена у сопротивления — риск отката выше. Оптимально — короткая позиция со стопом над зоной. При пробое и закреплении — сценарий пересмотреть. Дисциплина и стоп-лосс обязательны."],
        "neutral": ["Рынок в балансе — действую только по подтверждённому сигналу."]
    },
    "STOPLINE": [
        "Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа. Уровень выбран по волатильности для защиты капитала."
    ],
    "DISCLAIMER": "Это иллюстрация ИИ-идеи, а не инвестиционная рекомендация. Ситуация и волатильность могут резко меняться; прошлые результаты не гарантируют будущих. Торги сопряжены с высоким риском."
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
    # цветовые акценты управляются через CSS-классы
    cls = "arxora-card"
    if color == "green": cls += " entry"
    elif color == "red": cls += " sl"
    else: cls += " tp"
    return f"""
        <div class="{cls}">
            <div class="t">{title}</div>
            <div class="v">{value}</div>
            {f"<div class='s'>{sub}</div>" if sub else ""}
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

# Режим входа для шапки/карточки Entry (локальный fallback)
def entry_mode_labels(action: str, entry: float, last_price: float, eps: float):
    if action not in ("BUY", "SHORT"):
        return "WAIT", "Entry"
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
        st.markdown(f"<div class='arxora-h-num'>${last_price:.2f}</div>", unsafe_allow_html=True)

        action   = out["recommendation"]["action"]
        conf     = float(out["recommendation"].get("confidence", 0))
        conf_pct = f"{int(round(conf*100))}%"

        lv = dict(out["levels"])
        if action in ("BUY","SHORT"):
            t1,t2,t3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
            lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

        # --- шапка: Long/Short + Buy/Sell Stop/Limit/Market
        # Предпочтительно используем стратегию: entry_label / entry_kind
        mode_text = None
        if "entry_label" in out:
            # Нормализуем «NOW/STOP/LIMIT» в читабельный вид
            label = str(out["entry_label"])
            label = (label
                     .replace("NOW", "Market price")
                     .replace("STOP", "Stop")
                     .replace("LIMIT", "Limit"))
            mode_text = label.replace("Buy ", "Buy ").replace("Sell ", "Sell ")
        if not mode_text:
            mode_text, _entry_title_fallback = entry_mode_labels(
                action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS
            )

        header_text = "WAIT"
        if action == "BUY":
            header_text = f"Long • {mode_text}"
        elif action == "SHORT":
            header_text = f"Short • {mode_text}"

        st.markdown(
            f"""
            <div class="arxora-pill">
                <div style="font-size:1.15rem; font-weight:700;">{header_text}</div>
                <div style="opacity:0.8; font-size:0.95rem; margin-top:2px;">{conf_pct} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- карточки уровней
        if action in ("BUY", "SHORT"):
            # Заголовок для Entry — берем из entry_label, если есть; иначе из fallback
            if "entry_label" in out:
                entry_title = out["entry_label"].replace("NOW","Market").replace("STOP","Stop").replace("LIMIT","Limit")
            else:
                _, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2:
                st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            with c3:
                st.markdown(
                    card_html("TP 1", f"{lv['tp1']:.2f}",
                              sub=f"Probability {int(round(out['probs']['tp1']*100))}%"),
                    unsafe_allow_html=True,
                )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    card_html("TP 2", f"{lv['tp2']:.2f}",
                              sub=f"Probability {int(round(out['probs']['tp2']*100))}%"),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    card_html("TP 3", f"{lv['tp3']:.2f}",
                              sub=f"Probability {int(round(out['probs']['tp3']*100))}%"),
                    unsafe_allow_html=True,
                )

            rr = rr_line(lv)
            if rr:
                st.markdown(f"<div class='rr-line'>{rr}</div>", unsafe_allow_html=True)

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
        st.markdown(f"<div class='note' style='margin-top:8px'>{plan}</div>", unsafe_allow_html=True)

        ctx_key = "support" if action == "BUY" else ("resistance" if action == "SHORT" else "neutral")
        st.markdown(f"<div class='ctx'>{CUSTOM_PHRASES['CONTEXT'][ctx_key][0]}</div>", unsafe_allow_html=True)

        if action in ("BUY","SHORT"):
            stopline = CUSTOM_PHRASES["STOPLINE"][0].format(sl=_fmt(lv["sl"]), risk_pct=compute_risk_pct(lv))
            st.markdown(f"<div class='note' style='margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(
                f"<div class='note' style='margin-top:6px;'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>",
                unsafe_allow_html=True,
            )

        st.caption(f"<span class='disc'>{CUSTOM_PHRASES['DISCLAIMER']}</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать».")
