# app.py — Arxora (AI)
import os
import re
import hashlib
import random
import streamlit as st
from dotenv import load_dotenv

# Базовый движок сигналов (AI + правила)
from core.strategy import analyze_asset, analyze_asset_m7  # Добавлен импорт analyze_asset_m7

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
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))  # ~0.15%
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010"))

# ===================== ТЕКСТЫ =====================
CUSTOM_PHRASES = {
    "BUY": [
        "Точка входа: покупка в диапазоне {range_low}–{range_high}{unit_suffix}. По результатам AI-анализа выявлена ключевая область спроса в рамках текущего горизонта."
    ],
    "SHORT": [
        "Точка входа: продажа (short) в диапазоне {range_low}–{range_high}{unit_suffix}. AI-анализ выявил точку продаж в рамках текущего горизонта."
    ],
    "WAIT": [
        "Пока нет ясности с картиной на текущем горизонте, и я не спешу с решениями.",
        "Пока не стоит спешить — лучше дождаться более ясной картины. Вероятно, новости могут спровоцировать изменения на рынке.",
        "Пока без позиции — жду более чёткого сигнала. Новостной фон может сдвинуть рынок и повысить волатильность."
    ],
    "CONTEXT": {
        "support": ["Цена у уровня покупательской активности. Оптимальный вариант — открывать позицию на основе ордера, сгенерированного AI-анализом, рассчитанного на рост. Arxora учитывает множество факторов, которые трудно просчитать вручную, минимизируя эмоции и риски, и позволяя автоматически реагировать на быстро меняющиеся условия рынка. При этом важно контролировать риски: закрепление ниже этой зоны поддержки будет сигналом для пересмотра сценария. Торгуйте дисциплинированно. Строго соблюдайте уровни стоп-лосса."],
        "resistance": ["Цена подошла к сопротивлению, и вероятность коррекции здесь повышена. Оптимальный вариант — открывать позицию на основе ордера, сгенерированного AI-анализом, рассчитанного на падение. Такой подход обеспечивает максимально точный и своевременный вход в сделку. Arxora учитывает множество факторов, которые трудно просчитать вручную, минимизируя эмоции и риски, и позволяя автоматически реагировать на быстро меняющиеся условия рынка. При этом важно контролировать риски: закрепление выше зоны сопротивления будет сигналом для пересмотра сценария. Строго соблюдайте уровни стоп-лосса."],
        "neutral": ["Рынок пока в балансе — действую только по подтверждённому сигналу."]
    },
    "STOPLINE": [
        "Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа. Уровень определён алгоритмами анализа волатильности как критический для защиты капитала."
    ],
    "DISCLAIMER": "AI-анализ носит информационный характер, и не является прямой инвестиционной рекомендацией. Рыночная ситуация может быстро меняться, и Arxora адаптируется к этим изменениям. Прошлые результаты не гарантируют будущие. Торговля на финансовых рынках связана с высоким риском."
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

# Режим входа для шапки/карточки Entry
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
# Добавляем выбор стратегии
strategy_options = ["Основная стратегия", "M7 Strategy"]
selected_strategy = st.selectbox(
    "Стратегия",
    options=strategy_options,
    index=0,
    key="strategy_select"
)

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
st.write(f"Mode: {'AI (pseudo)' if AI_PSEUDO else 'AI'} · Horizon: {hz_tag} · Strategy: {selected_strategy}")

# ===================== Main =====================
if run and ticker:
    try:
        # Выбираем стратегию в зависимости от выбора
        if selected_strategy == "M7 Strategy":
            out = analyze_asset_m7(ticker=symbol_for_engine, horizon=horizon)
        else:
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

        # --- шапка: Long/Short + Buy/Sell Stop/Limit/Market
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

        # --- карточки уровней
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
            # ⬇️ СДЕЛАНО: RR теперь оранжевым
            if rr:
                st.markdown(
                    f"<div style='margin-top:4px; color:#FFA94D; font-weight:600;'>{rr}</div>",
                    unsafe_allow_html=True,
                )

        # --- план/контекстст/стоп-/стоп-линия
        def render_planлиния
        def render_plan_line(action, levels, ticker="", seed_extra=""):
            seed = int(hashlib.sha1(f"{ticker}{seed_extra}{levels['entry']}{levels['sl']}{action}".encode()).hexdigest(), 16) % (2**32)
            rnd = random.Random(seed)
            if action == "WAIT":
                return rnd.choice(CUSTOM_PHRASES["WAIT"])
            rng_low, rng_high = compute_display_range(_line(action, levels, ticker="", seed_extra=""):
            seed = int(hashlib.sha1(f"{ticker}{seed_extra}{levels['entry']}{levels['sl']}{action}".encode()).hexdigest(), 16) % (2**32)
            rnd = random.Random(seed)
            if action == "WAIT":
                return rnd.choice(CUSTOM_PHRASES["WAIT"])
            rng_low, rng_high = compute_display_range(levels)
           levels)
            us = unit_suffix(ticker)
            tpl = CUSTOM_PHRASES[action][ us = unit_suffix(ticker)
            tpl = CUSTOM_PHRASES[action][0]
            return t0]
            return tpl.formatpl.format(range_low(range_low=rng_low, range_=rng_low, range_high=rhigh=rng_ng_high, unit_suffix=us)

        plan = render_plan_line(action, lv, ticker=ticker, seed_extra=horizon)
        st.markdown(f"<div style='margin-top:8px'>{plan}</high, unit_suffix=us)

        plan = render_plan_line(action, lv, ticker=ticker, seed_extra=horizon)
        st.markdown(f"<div style='margin-top:8px'>{plan}</divdiv>", unsafe_allow_html=True)

>", unsafe_allow_html=True)

        ctx        ctx_key = "_key = "supportsupport" if action ==" if action == "BUY" "BUY" else ("resistance else ("resistance" if" if action == "SH action == "SHORT" else "ORT" else "neutral")
        stneutral")
        st.markdown.markdown(f"<(f"<div style='opacitydiv style='opacity:0.9':0.9'>{CUSTOM_P>{CUSTOM_PHRASES['CONTEXTHRASES['CONTEXT'][ctx_key][0'][ctx_key][0]}</div>",]}</div>", unsafe_allow_html unsafe_allow_html=True)

        if action in=True)

        if action in ("BUY"," ("BUY","SHSHORT"):
           ORT"):
            stopline = CUSTOM_P stopline = CUSTOM_PHRASES["STHRASES["STOPLINE"OPLINE"][0].format(s][0].format(sl=_fmt(ll=_fmt(lv["v["sl"]), risksl"]), risk_p_pct=compute_risk_pctct=compute_risk_pct(lv(lv))
            st.mark))
            st.markdown(f"down(f"<div<div style style='opacity:0.9; margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(
                f"<div style='margin-top:6px;'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>",
               ='opacity:0.9; margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(
                f"<div style='margin-top:6px;'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>",
                unsafe_allow_html=True,
            unsafe_allow_html=True,
            )

 )

        st.caption        st.caption(CUSTOM(CUSTOM_PHRASES["_PHRASES["DISCLAIMERDISCLAIMER"])

"])

    except Exception    except Exception as as e:
        e:
        st st.error(f"Ошиб.error(f"Ошибкака анализа: {e анализа: {e}")
elif not tick}")
elif not ticker:
er:
    st.info    st.info("Введите("Введите тик тикер иер и наж нажмите «мите «ПроаПроаналинализировать»зировать».")

# =.")

# ========================================= НИЖНИ НИЖНИЙЙ К КОЛОЛОНТИТУОНТИТУЛ =================Л =====================
st.mark====
st.markdown("---down("---")

# Добавля")

# Добавляем CSS для жирности кнопок
st.markdownем CSS для жирности кнопок
st.markdown("""
<style>
    .("""
<style>
    .ststButton > button {
Button > button {
               font-weight:  font-weight: 600;
600;
    }
    }
</style>
""</style>
""",", unsafe_allow unsafe_allow_html=True_html=True)

# Соз)

# Создаем центридаем центрированные крованные кнопки
нопки
colcol1, col21, col2, col3,, col3, col col44, col5 =, col5 = st.columns st.columns([1, 1,([1, 1, 2, 1, 2, 1, 1])

with 1])

with col2 col2:
   :
    if st if st.button("Arx.button("Arxora", use_containerora", use_container_width_width=True):
        st=True):
        st.session.session_state.show_ar_state.show_arxora = notxora = not st.session st.session_state.get('show_state.get('show__arxoraarxora', False', False)
        st)
        st.session.session_state.show_crypto_state.show_crypto = False = False

with

with col3 col3:
    st:
    st.button(".button("US Stocks",US Stocks", use_container use_container_width=True_width=True)

with)

with col4:
    col4:
    if st.button if st.button("("CCrypto", use_container_width=True):
        st.session_staterypto", use_container_width=True):
        st.session_state.show_crypto.show_crypto = not st.session_state.get = not st.session_state.get('show_crypto', False)
        st.session_state.show_('show_crypto', False)
        st.session_state.show_arxoraarxora = False

 = False

# Отобра# Отображажаем информацию при необходимостием информацию при необходимости
if
if st.session st.session_state_state.get('.get('show_arshow_arxoraxora', False):
    st.mark', False):
    st.markdown(
down(
        """
               """
        < <div style="backgrounddiv style="background-color-color: #000: #000000;000; color: #ffffff color: #ffffff; padding; padding: : 15px15px; border; border-radius: 10px; margin-top-radius: 10px; margin-top: 10px: 10px;">
            <h;">
            <h44 style="font-weight: style="font-weight: 600;"> 600;">ОО проекте</h4 проекте</h4>
            <p style="font-weight:>
            <p style="font-weight: 300 300;">
            Arxora AI — это современное;">
            Arxora AI — это современное решение, которое помогает тре решение, которое помогает трейдерамйдерам принимать точные принимать точные и об и обосноваоснованные решениянные решения 
            
            на финансовых на финансовых рынках рынках с помощью перед с помощью передовых технологийовых технологий искусственного интеллек искусственного интеллекта и машинта и машинного обучения. 
           ного обучения. 
            Arxora помогает трейдера Arxora помогает трейдерам автоматизим автоматизировать анализровать анализ, повы, повышатьшать качество качество входов и управ входов и управлять рисками,лять рисками, 
            дела 
            делаяя торговлю проще, эффективнее и торговлю проще, эффективнее и раз разумнее. Бумнее. Благлагодаря высокой скорости обработодаря высокой скорости обработки данных Arxoraки данных Arxora может быстро предоставить анализ может быстро предоставить анализ большого количества активов большого количества активов за очень короткое время. Это уп за очень короткое время. Это упрощает торговлю, позволярощает торговлю, позволяя трейя трейдерадерам легко осуществлятьм легко осуществлять самопроверку и рассматри самопроверку и рассматривать альтернативныевать альтерна варианты решений.тивные варианты решений. Ключевые Ключевые особенности платфор особенности платформы: AI Override —мы: AI Over это встроенныйride — это встроенный механизм, который позволяет механизм, который позволяет искусственному инте искусственному интелллекту вмешилекту вмешиваться в работу базовыхваться в работу баз алгоритмов и принимать болееовых алгоритмов и принимать более точные решения в моменты точные решения в моменты, когда, когда рынок рынок вед ведёт себя нестандартно.
           ёт себя нестандартно Вероятностный анализ: Исп.
            Вероятностный анализ: Используяользуя мощные алгоритмы машин мощные алгоритмы машинногоного обучения, обучения, система рассчитывает вероятность успеха каждой сделки и присваивает уровень confidence ( система рассчитывает вероятность успеха каждой сделки и присваивает уровень confidence (%),%), что дает прозрачность что дает прозрачность и помогает управлять риска и помогает управлять рисками.
            Машинное обучение (ML): Система постоянно обучается на исторических данных и поведении рынка, совершенствуя модели и адаптируясь к изменениям рыночнойми.
            Машинное обучение (ML): Система постоянно обучается на исторических данных и поведении рынка, совершенствуя модели и адаптируясь к изменениям рыночной конъюнктуры конъюнктуры. Попробуйте мощь искусственного интеллекта в трейдинге уже сегодня!
           . Попробуйте мощь искусственного интеллекта в трейдинге уже сегодня!
            </p </p>
        </>
        </divdiv>
        """,
>
        """,
        unsafe        unsafe_allow_allow_html=True_html=True
    )


    )

ifif st.session_state.get('show_crypto st.session_state.get('show_crypto', False):
    st', False):
   .markdown(
        """
 st.markdown(
        """
        <div style        <div style="background-color: #000="background-color: #000000; color:000; color: #ffffff; padding #ffffff; padding: 15px: 15px; border-radius: 10; border-radius: 10px; margin-toppx; margin-top: 10px: 10px;">
            <h4 style="font-weight: ;">
            <h4 style="font-weight: 600600;">Crypto</;">Crypto</h4>
           h4>
            < <p style="font-weightp style="font-weight: 300;">
            Arx: 300;">
            Arxora анализируетora анализирует основные крип основные криптовалтовалюты 
            (юты 
            (Bitcoin, EthereumBitcoin, Ethereum и другие и другие) с использованием) с использованием тех тех же алгоритмических же алгоритмических подходов, что подходов, что и и для традиционных для традиционных активов.
            </p активов.
            </p>
            <>
            <p style="font-weight: 500p style="font-weight: 500;">Особ;">Особенности крипто-аенности крипто-анализа:</p>
           нализа:</p>
            <ul style <ul style="font="font-weight-weight: 350: 350;">
                <li;">
                <li>У>Учетчет высокой в высокой волатильолатильности криности криптовалютптовалют</li</li>
                <li>
                <li>Анализ>Анализ круглос круглосуточуточного рынного рынка</ка</li>
               li>
                <li> <li>УчетУчет специфи специфических крических крипто-фпто-факторовакторов</li</li>
           >
            </ul>
 </ul>
               </div>
        """,
        unsafe </div>
        """,
        unsafe_allow_allow_html=True
   _html=True
    )
