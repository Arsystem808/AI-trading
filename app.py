# app.py
import os
import re
import hashlib
import random
import streamlit as st
from dotenv import load_dotenv
from core.strategy import analyze_asset

# =====================
# Page config (ставим самым первым Streamlit-вызовом)
# =====================
st.set_page_config(
    page_title="Arxora — трейд-ИИ (MVP)",
    page_icon="assets/arxora_favicon_512.png",
    layout="centered"
)

# =====================
# Env: .env + Streamlit secrets -> os.environ
# =====================
load_dotenv()  # подхват из .env, если есть
for k in ("POLYGON_API_KEY", "ARXORA_MODEL_DIR",
          "ARXORA_AI_TH_LONG", "ARXORA_AI_TH_SHORT",
          "ARXORA_AI_PSEUDO"):
    if k in st.secrets:
        os.environ[k] = str(st.secrets[k])

# =====================
# Arxora BRANDING
# =====================
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
            """, unsafe_allow_html=True,
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
def _fmt(x):
    return f"{float(x):.2f}"

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

# =====================
# HTML карточка (используем в колонках)
# =====================
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
# Polygon нормализация (всегда Polygon)
# =====================
def normalize_for_polygon(symbol: str) -> str:
    """
    Возвращает тикер в формате Polygon.
    Примеры:
      'X:btcusd' -> 'X:BTCUSD'
      'BTCUSDT'  -> 'X:BTCUSD'
      'ETHUSD'   -> 'X:ETHUSD'
      'AAPL'     -> 'AAPL' (акции/ETF)
    """
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
col1, col2 = st.columns([2,1])
with col1:
    ticker_input = st.text_input(
        "Тикер",
        value="",  # без AAPL по умолчанию
        placeholder="Примеры: AAPL · TSLA · X:BTCUSD · BTCUSDT"
    )
    ticker = ticker_input.strip().upper()
with col2:
    horizon = st.selectbox(
        "Горизонт",
        ["Краткосрок (1–5 дней)", "Среднесрок (1–4 недели)", "Долгосрок (1–6 месяцев)"],
        index=1
    )

# --- AI mode badge ---
hz = "ST" if "Кратко" in horizon else ("MID" if "Средне" in horizon else "LT")
model_dir = os.getenv("ARXORA_MODEL_DIR", "models")
model_path = os.path.join(model_dir, f"arxora_lgbm_{hz}.joblib")
ai_has_model = os.path.exists(model_path)
pseudo_on = str(os.getenv("ARXORA_AI_PSEUDO", "1")).lower() in ("1", "true", "yes")
mode_label = "AI" if ai_has_model else ("AI (pseudo)" if pseudo_on else "Rules")
st.markdown(
    f"<div style='opacity:.75;font-size:.9rem;margin:.25rem 0'>Mode: <b>{mode_label}</b> · Horizon: {hz}</div>",
    unsafe_allow_html=True
)

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
        conf = out["recommendation"].get("confidence", 0)
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
                st.markdown(card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out['probs']['tp1']*100))}%"), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out['probs']['tp2']*100))}%"), unsafe_allow_html=True)
            with c2:
                st.markdown(card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out['probs']['tp3']*100))}%"), unsafe_allow_html=True)

            rr = rr_line(lv)
            if rr:
                st.markdown(f"<div style='opacity:0.75; margin-top:4px'>{rr}</div>", unsafe_allow_html=True)

        plan = render_plan_line(action, lv, ticker=ticker, seed_extra=horizon)
        st.markdown(f"<div style='margin-top:8px'>{plan}</div>", unsafe_allow_html=True)

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
# Admin · Train ST model inline (optional)
# =====================
with st.expander("🧠 ML · быстрый тренинг (ST) прямо здесь"):
    st.caption("Обучит краткосрочную модель (ST) по дневным данным из Polygon и сохранит её в models/.")

    tickers_text = st.text_input(
        "Тикеры (через запятую)",
        "SPY,QQQ,AAPL,MSFT,TSLA,X:BTCUSD,X:ETHUSD",
        key="train_tickers"
    )
    months = st.slider("Месяцев истории", 3, 36, 12, key="train_months")
    start_btn = st.button("🚀 Обучить ST-модель сейчас", key="train_start")

    if start_btn:
        try:
            import joblib
            # LightGBM опционален, fallback — LogisticRegression
            try:
                from lightgbm import LGBMClassifier
                use_lgbm = True
            except Exception:
                from sklearn.linear_model import LogisticRegression
                use_lgbm = False
        except Exception:
            st.error("Нужно установить зависимости: `pip install joblib scikit-learn lightgbm`")
            st.stop()

        from pathlib import Path
        import numpy as np
        import pandas as pd

        # берем нужные хелперы из твоей стратегии, чтобы совпали фичи/логика
        try:
            from core.polygon_client import PolygonClient
            from core.strategy import (
                _horizon_cfg, _atr_like, _weekly_atr, _linreg_slope, _streak,
                _last_period_hlc, _fib_pivots, _classify_band,
                _apply_tp_floors, _order_targets, _hz_tag, _three_targets_from_pivots
            )
        except Exception as e:
            st.error(f"Импорт модулей не удался: {e}")
            st.stop()

        HORIZON = "Краткосрок (1–5 дней)"
        cfg = _horizon_cfg(HORIZON)
        look = cfg["look"]
        HOLD_DAYS = 5
        FILL_WINDOW = 3
        FEATS = ["pos","slope_norm","atr_d_over_price","vol_ratio","streak","band","long_upper","long_lower"]

        def compute_levels_asof(df_asof: pd.DataFrame):
            price = float(df_asof["close"].iloc[-1])
            atr_d  = float(_atr_like(df_asof, n=cfg["atr"]).iloc[-1])
            atr_w  = _weekly_atr(df_asof) if cfg.get("use_weekly_atr") else atr_d
            hlc = _last_period_hlc(df_asof, cfg["pivot_period"])
            if not hlc:
                hlc = (float(df_asof["high"].tail(60).max()),
                       float(df_asof["low"].tail(60).min()),
                       float(df_asof["close"].iloc[-1]))
            H,L,C = hlc
            piv = _fib_pivots(H,L,C)
            P,R1,R2 = piv["P"],piv["R1"],piv.get("R2")
            step_w = atr_w
            if price < P:
                entry = max(price, piv["S1"] + 0.15*step_w); sl = piv["S1"] - 0.60*step_w
            else:
                entry = max(price, P + 0.10*step_w); sl = P - 0.60*step_w
            hz = _hz_tag(HORIZON)
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, "BUY", hz, price, step_w)
            tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, "BUY")
            return entry, sl, tp1

        def label_tp_before_sl(df: pd.DataFrame, start_ix: int, entry: float, sl: float, tp1: float, hold_days: int) -> int:
            lo = df["low"].values; hi = df["high"].values
            N = len(df)
            end = min(N, start_ix + hold_days)
            for k in range(start_ix, end):
                if lo[k] <= sl:  return 0
                if hi[k] >= tp1: return 1
            return 0

        cli = PolygonClient()
        rows = []
        tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

        prog = st.progress(0.0, text="Готовим датасет…")
        total = len(tickers)

        for idx, t in enumerate(tickers, 1):
            try:
                days = int(months*31) + look + 40
                df = cli.daily_ohlc(t, days=days).dropna()
                if len(df) < look + 30:
                    st.write(f"⚠️ {t}: мало данных ({len(df)}) — пропускаю")
                    prog.progress(min(idx/total, 1.0))
                    continue

                for i in range(look+5, len(df)-6):
                    df_asof = df.iloc[:i+1]
                    price = float(df_asof["close"].iloc[-1])

                    tail = df_asof.tail(look)
                    rng_low, rng_high = float(tail["low"].min()), float(tail["high"].max())
                    rng_w = max(1e-9, rng_high - rng_low)
                    pos = (price - rng_low) / rng_w

                    atr_d = float(_atr_like(df_asof, n=cfg["atr"]).iloc[-1])
                    atr2  = float(_atr_like(df_asof, n=cfg["atr"]*2).iloc[-1])
                    vol_ratio = atr_d/max(1e-9, atr2)
                    slope  = _linreg_slope(df_asof["close"].tail(cfg["trend"]).values)/max(1e-9, price)
                    streak = _streak(df_asof["close"])

                    hlc = _last_period_hlc(df_asof, cfg["pivot_period"])
                    if not hlc:
                        hlc = (float(df_asof["high"].tail(60).max()),
                               float(df_asof["low"].tail(60).min()),
                               float(df_asof["close"].iloc[-1]))
                    piv = _fib_pivots(*hlc)
                    band = _classify_band(price, piv, 0.25*(_weekly_atr(df_asof) if cfg.get("use_weekly_atr") else atr_d))

                    lw_row = df_asof.iloc[-1]
                    body  = abs(lw_row["close"] - lw_row["open"])
                    upper = max(0.0, lw_row["high"] - max(lw_row["open"], lw_row["close"]))
                    lower = max(0.0, min(lw_row["open"], lw_row["close"]) - lw_row["low"])
                    long_upper = (upper > body*1.3) and (upper > lower*1.1)
                    long_lower = (lower > body*1.3) and (lower > upper*1.1)

                    entry, sl, tp1 = compute_levels_asof(df_asof)

                    touch_ix = None
                    j_end = min(i + 1 + FILL_WINDOW, len(df))
                    for j in range(i+1, j_end):
                        lo = float(df["low"].iloc[j]); hi = float(df["high"].iloc[j])
                        if lo <= entry <= hi:
                            touch_ix = j
                            break
                    if touch_ix is None:
                        continue

                    y = label_tp_before_sl(df, touch_ix, entry, sl, tp1, HOLD_DAYS)

                    rows.append(dict(
                        ticker=t, date=df_asof.index[-1].date(), y=int(y),
                        pos=pos, slope_norm=slope, atr_d_over_price=atr_d/max(1e-9, price),
                        vol_ratio=vol_ratio, streak=float(streak), band=float(band),
                        long_upper=float(long_upper), long_lower=float(long_lower)
                    ))

            except Exception as e:
                st.write(f"❌ {t}: {e}")
            finally:
                prog.progress(min(idx/total, 1.0))

        if not rows:
            st.error("Не удалось собрать датасет. Проверь ключ Polygon/тикеры/историю.")
            st.stop()

        dfX = pd.DataFrame(rows)
        st.write("📊 Размер датасета:", dfX.shape, "доля y=1:", float(dfX["y"].mean()))

        X = dfX[FEATS].astype(float); y = dfX["y"].astype(int)

        if use_lgbm:
            model = LGBMClassifier(
                n_estimators=400, learning_rate=0.05,
                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                random_state=42
            )
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=500, class_weight="balanced")

        model.fit(X, y)
        # упаковка: как в ai_inference.load_model ожидается {model, features, auc}
        from sklearn.metrics import roc_auc_score
        try:
            proba = model.predict_proba(X)[:,1]
        except Exception:
            proba = model.predict(X).astype(float)
        auc = float(roc_auc_score(y, proba)) if len(np.unique(y))>1 else float("nan")

        models_dir = Path(os.getenv("ARXORA_MODEL_DIR","models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / "arxora_lgbm_ST.joblib"
        joblib.dump({"model": model, "features": FEATS, "auc": auc}, out_path)
        
with open(out_path, "rb") as f:
    st.download_button("⬇️ Скачать модель (ST)",
                       data=f.read(),
                       file_name="arxora_lgbm_ST.joblib")

        st.success(f"✅ Модель сохранена: {out_path}")
        st.write(f"AUC по обучению (грубо): {auc:.3f}")
        st.info("Перезапусти анализ (или приложение), бейдж должен переключиться на Mode: AI.")

