# app.py
import os
import re
import hashlib
import random
import importlib
import streamlit as st
from dotenv import load_dotenv

# ====== core logic ======
from core.strategy import analyze_asset

load_dotenv()

# ---------------- Env toggles ----------------
MODEL_DIR         = os.getenv("ARXORA_MODEL_DIR", "models")
AI_PSEUDO         = os.getenv("ARXORA_AI_PSEUDO", "0") == "1"
AI_TH_LONG        = float(os.getenv("ARXORA_AI_TH_LONG",  "0.60"))
AI_TH_SHORT       = float(os.getenv("ARXORA_AI_TH_SHORT", "0.60"))
SHOW_TRAINERS     = os.getenv("ARXORA_SHOW_TRAINERS", "0") == "1"
TRAINER_PASS      = os.getenv("ARXORA_TRAINER_PASS", "admin")

# ------------- try import trainers (optional) -------------
HAVE_TRAINERS = False
train_quick_st = train_quick_mid = train_quick_lt = None
try:
    ai_inf_mod = importlib.import_module("core.ai_inference")
    train_quick_st  = getattr(ai_inf_mod, "train_quick_st",  None)
    train_quick_mid = getattr(ai_inf_mod, "train_quick_mid", None)
    train_quick_lt  = getattr(ai_inf_mod, "train_quick_lt",  None)
    HAVE_TRAINERS = all([train_quick_st, train_quick_mid, train_quick_lt])
except Exception:
    HAVE_TRAINERS = False

# ========== page config / branding ==========
st.set_page_config(page_title="Arxora — трейд-ИИ (MVP)",
                   page_icon="assets/arxora_favicon_512.png",
                   layout="centered")

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

# =====================
# Полезные фразы
# =====================
CUSTOM_PHRASES = {
    "BUY": [
        "Точка входа: покупка в диапазоне {range_low}–{range_high}{unit_suffix}. AI-анализ указывает на сильную поддержку в этой зоне."
    ],
    "SHORT": [
        "Точка входа: продажа (short) в диапазоне {range_low}–{range_high}{unit_suffix}. AI-анализ указывает на сильное сопротивление в этой зоне."
    ],
    "WAIT": [
        "Пока не вижу для себя ясной картины, я бы не торопился.",
        "Я бы пока не торопился и подождал более точной картины. Возможные новости могут стать триггером и изменить динамику и волатильность.",
        "Пока без позиции: жду более ясного сигнала. Новости могут сдвинуть рынок и поменять волатильность."
    ],
    "CONTEXT": {
        "support": ["Анализ, проведённый ИИ, определяет эту зону как область сильной поддержки."],
        "resistance": ["Анализ, проведённый ИИ, определяет эту зону как область сильного сопротивления."],
        "neutral": ["Рынок в балансе — действую только по подтверждённому сигналу."]
    },
    "STOPLINE": [
        "Стоп-лосс: {sl}. Потенциальный риск ~{risk_pct}% от входа. Уровень определён алгоритмами анализа волатильности как критический."
    ],
    "DISCLAIMER": "Данная информация является примером того, как AI может генерировать инвестиционные идеи и не является прямой инвестиционной рекомендацией. Торговля на финансовых рынках сопряжена с высоким риском."
}

# =====================
# Хелперы (формат/риск/юниты)
# =====================
def _fmt(x): return f"{float(x):.2f}"

def compute_display_range(levels, widen_factor=0.25):
    entry = float(levels["entry"]); sl = float(levels["sl"])
    risk = abs(entry - sl)
    width = max(risk * widen_factor, 0.01)
    low, high = entry - width, entry + width
    return _fmt(min(low, high)), _fmt(max(low, high))

def compute_risk_pct(levels):
    entry = float(levels["entry"]); sl = float(levels["sl"])
    return "—" if entry == 0 else f"{abs(entry - sl)/abs(entry)*100.0:.1f}"

UNIT_STYLE = {"equity":"za_akciyu","etf":"omit","crypto":"per_base","fx":"per_base","option":"per_contract"}
ETF_HINTS = {"SPY","QQQ","IWM","DIA","EEM","EFA","XLK","XLF","XLE","XLY","XLI","XLV","XLP","XLU","VNQ","GLD","SLV"}

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
        if t.endswith(q) and len(t) > len(q):
            return t[:-len(q)]
    return re.split(r"[-:/]", ticker.upper())[0].replace("X:","").replace("C:","")

def unit_suffix(ticker: str) -> str:
    kind = detect_asset_class(ticker)
    style = UNIT_STYLE.get(kind, "omit")
    if style == "za_akciyu":    return " за акцию"
    if style == "per_base":      return f" за 1 {parse_base_symbol(ticker)}"
    if style == "per_contract":  return " за контракт"
    return ""

def rr_line(levels):
    risk = abs(levels["entry"] - levels["sl"])
    if risk <= 1e-9: return ""
    rr1 = abs(levels["tp1"] - levels["entry"]) / risk
    rr2 = abs(levels["tp2"] - levels["entry"]) / risk
    rr3 = abs(levels["tp3"] - levels["entry"]) / risk
    return f"RR ≈ 1:{rr1:.1f} (TP1) · 1:{rr2:.1f} (TP2) · 1:{rr3:.1f} (TP3)"

def render_plan_line(action, levels, ticker="", seed_extra=""):
    seed = int(hashlib.sha1(f"{ticker}{seed_extra}{levels['entry']}{levels['sl']}{action}".encode()).hexdigest(), 16) % (2**32)
    rnd = random.Random(seed)
    if action == "WAIT":
        return rnd.choice(CUSTOM_PHRASES["WAIT"])
    rng_low, rng_high = compute_display_range(levels)
    us = unit_suffix(ticker)
    tpl = CUSTOM_PHRASES[action][0]
    return tpl.format(range_low=rng_low, range_high=rng_high, unit_suffix=us)

def render_context_line(kind_key="neutral"):
    return CUSTOM_PHRASES["CONTEXT"].get(kind_key, CUSTOM_PHRASES["CONTEXT"]["neutral"])[0]

def render_stopline(levels):
    line = CUSTOM_PHRASES["STOPLINE"][0]
    return line.format(sl=_fmt(levels["sl"]), risk_pct=compute_risk_pct(levels))

def card_html(title, value, sub=None, color=None):
    bg = "#141a20"
    if color == "green": bg = "#123b2a"
    elif color == "red": bg = "#3b1f20"
    return f"""
        <div style="background:{bg}; padding:12px 16px; border-radius:14px; border:1px solid rgba(255,255,255,0.06); margin:6px 0;">
            <div style="font-size:0.9rem; opacity:0.85;">{title}</div>
            <div style="font-size:1.4rem; font-weight:700; margin-top:4px;">{value}</div>
            {f"<div style='font-size:0.8rem; opacity:0.7; margin-top:2px;'>{sub}</div>" if sub else ""}
        </div>
    """

# =====================
# Polygon нормализация
# =====================
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

# =====================
# Inputs
# =====================
HZ_LABELS = {
    "Краткосрок (1–5 дней)": "ST",
    "Среднесрок (1–4 недели)": "MID",
    "Долгосрок (1–6 месяцев)": "LT",
}

col1, col2 = st.columns([2,1])
with col1:
    ticker_input = st.text_input(
        "Тикер",
        value="",
        placeholder="Примеры: AAPL · TSLA · X:BTCUSD · BTCUSDT"
    )
    ticker = ticker_input.strip().upper()
with col2:
    horizon = st.selectbox(
        "Горизонт",
        list(HZ_LABELS.keys()),
        index=1
    )

ai_mode_label = "AI (pseudo)" if AI_PSEUDO else "AI"
st.markdown(f"<div style='opacity:0.9;'>Mode: <b>{ai_mode_label}</b> · Horizon: <b>{HZ_LABELS[horizon]}</b></div>", unsafe_allow_html=True)

symbol_for_engine = normalize_for_polygon(ticker)
run = st.button("Проанализировать", type="primary")

# =====================
# Main
# =====================
if run:
    try:
        out = analyze_asset(ticker=symbol_for_engine, horizon=horizon)

        st.markdown(
            f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>${out['last_price']:.2f}</div>",
            unsafe_allow_html=True,
        )

        action = out["recommendation"]["action"]
        conf = out["recommendation"].get("confidence", 0.0)
        conf_pct = f"{int(round(float(conf)*100))}%"
        action_text = "Buy LONG" if action == "BUY" else ("Sell SHORT" if action == "SHORT" else "WAIT")
        st.markdown(
            f"""
            <div style="background:#0f1b2b; padding:14px 16px; border-radius:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:10px;">
                <div style="font-size:1.15rem; font-weight:700;">{action_text}</div>
                <div style="opacity:0.75; font-size:0.95rem; margin-top:2px;">{conf_pct} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        lv = out["levels"]
        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(card_html("Entry", f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
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
                # <<< fixed the typo here: unsafe_allow_html >>>
                st.markdown(
                    card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out['probs']['tp3']*100))}%"),
                    unsafe_allow_html=True
                )

            rr = rr_line(lv)
            if rr:
                st.markdown(f"<div style='opacity:0.75; margin-top:4px'>{rr}</div>", unsafe_allow_html=True)

        # Text plan & context
        def _fmt_line(html): st.markdown(f"<div style='margin-top:8px'>{html}</div>", unsafe_allow_html=True)

        plan = render_plan_line(action, lv, ticker=ticker, seed_extra=horizon)
        _fmt_line(plan)

        ctx_key = "neutral"
        if action == "BUY": ctx_key = "support"
        elif action == "SHORT": ctx_key = "resistance"
        ctx = render_context_line(ctx_key)
        st.markdown(f"<div style='opacity:0.9'>{ctx}</div>", unsafe_allow_html=True)

        if action in ("BUY","SHORT"):
            stopline = render_stopline(lv)
            st.markdown(f"<div style='opacity:0.9; margin-top:4px'>{stopline}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(f"<div style='margin-top:6px;'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>", unsafe_allow_html=True)

        st.caption(CUSTOM_PHRASES["DISCLAIMER"])

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
else:
    st.info("Введите тикер и нажмите «Проанализировать».")

# =====================
# ML панель (PIN + тренажёры)
# =====================
if "ml_unlocked" not in st.session_state:
    st.session_state.ml_unlocked = False

def trainer_block(title, trainer_fn):
    with st.expander(title, expanded=False):
        if not HAVE_TRAINERS or trainer_fn is None:
            st.warning("Тренажёры недоступны: не найдены функции train_quick_st/mid/lt в core/ai_inference.py. "
                       "Используйте готовые модели в каталоге models/.", icon="⚠️")
            return
        tickers = st.text_input("Тикеры (через запятую)", value="aapl")
        months  = st.slider("Месяцев истории", min_value=6, max_value=48, value=18, step=1)
        if st.button("🚀 Обучить модель сейчас"):
            try:
                out_path, auc, shape, pos_share = trainer_fn(tickers, months, MODEL_DIR)
                st.success(f"✅ Модель сохранена: {out_path}")
                st.markdown(f"**Размер датасета:** {shape} · **доля y=1:** {pos_share:.4f}")
                st.markdown(f"AUC по обучению (грубо): **{auc:.3f}**")
                # Если тренер возвращает bytes — можно дать скачать. Иначе пропускаем.
                try:
                    with open(out_path, "rb") as fh:
                        st.download_button("⬇️ Скачать модель", data=fh, file_name=os.path.basename(out_path))
                except Exception:
                    pass
            except Exception as ex:
                st.error(f"Ошибка обучения: {ex}")

if SHOW_TRAINERS or st.session_state.ml_unlocked:
    trainer_block("🧠 ML · быстрый тренинг (ST) прямо здесь",  train_quick_st)
    trainer_block("🧠 ML · быстрый тренинг (MID) прямо здесь", train_quick_mid)
    trainer_block("🧠 ML · быстрый тренинг (LT) прямо здесь",  train_quick_lt)
else:
    with st.expander("🔐 Открыть ML-панели (PIN)", expanded=False):
        pin = st.text_input("Введите PIN для доступа к ML-панелям", type="password")
        if st.button("Открыть"):
            if pin == TRAINER_PASS:
                st.session_state.ml_unlocked = True
                st.success("Доступ открыт. Прокрутите страницу — панели появились.")
            else:
                st.error("Неверный PIN.")

