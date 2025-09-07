# app.py
import os
import re
import hashlib
import random
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.strategy import analyze_asset           # ваш основной движок (AI + правила)
from core.polygon_client import PolygonClient     # для бэктеста берём котировки

load_dotenv()

# =========================================================
# BRANDING
# =========================================================
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
            """, unsafe_allow_html=True,
        )

render_arxora_header()

# =========================================================
# ТЕКСТЫ
# =========================================================
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
    "DISCLAIMER": "Данная информация является примером того, как AI может генерировать инвестиционные идеи и не является прямой инвестиционной рекомендацией. Рыночная конъюнктура может быстро меняться; прошлые результаты не гарантируют будущих. Торговля на финансовых рынках сопряжена с высоким риском."
}

# =========================================================
# HELPER'ы (formatter/юниты/карточки)
# =========================================================
def _fmt(x): return f"{float(x):.2f}"

def compute_display_range(levels, widen_factor=0.25):
    entry = float(levels["entry"]); sl = float(levels["sl"])
    risk = abs(entry - sl); width = max(risk * widen_factor, 0.01)
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
        if t.endswith(q) and len(t) > len(q): return t[:-len(q)]
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

# =========================================================
# Нормализация тикера под Polygon
# =========================================================
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

# =========================================================
# INPUTS
# =========================================================
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

# =========================================================
# MAIN: анализ одного актива
# =========================================================
if run and ticker:
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
            with c1: st.markdown(card_html("Entry", f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            with c3: st.markdown(card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out['probs']['tp1']*100))}%"), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1: st.markdown(card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out['probs']['tp2']*100))}%"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out['probs']['tp3']*100))}%"), unsafe_allow_html=True)

            rr = rr_line(lv)
            if rr: st.markdown(f"<div style='opacity:0.75; margin-top:4px'>{rr}</div>", unsafe_allow_html=True)

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
elif not ticker:
    st.info("Введите тикер и нажмите «Проанализировать».")

# =========================================================
# БЭКТЕСТ: без дублей + 3 политики выхода
# =========================================================
# --- тех. функции (ATR/линии, мини-логика для бэктеста) ---
def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _weekly_atr(df: pd.DataFrame, n_weeks: int = 8) -> float:
    w = df.resample("W-FRI").agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(w) < 2: return float((df["high"] - df["low"]).tail(14).mean())
    hl = w["high"] - w["low"]
    hc = (w["high"] - w["close"].shift(1)).abs()
    lc = (w["low"] - w["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.rolling(n_weeks, min_periods=1).mean().iloc[-1])

def _linreg_slope(y: np.ndarray) -> float:
    n = len(y); 
    if n < 2: return 0.0
    x = np.arange(n, dtype=float); xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0: return 0.0
    beta = ((x - xm) * (y - ym)).sum() / denom
    return float(beta)

def _streak(closes: pd.Series) -> int:
    s = 0
    for i in range(len(closes)-1, 0, -1):
        d = closes.iloc[i] - closes.iloc[i-1]
        if d > 0:
            if s < 0: break
            s += 1
        elif d < 0:
            if s > 0: break
            s -= 1
        else:
            break
    return s

def _wick_profile(row):
    o, c, h, l = row["open"], row["close"], row["high"], row["low"]
    body = abs(c - o)
    upper = max(0.0, h - max(o, c))
    lower = max(0.0, min(o, c) - l)
    return body, upper, lower

def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(g) < 2: return None
    row = g.iloc[-2]
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
    S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
    return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3}

def _classify_band(price: float, piv: dict, buf: float) -> int:
    P, R1 = piv["P"], piv["R1"]
    R2, R3, S1, S2 = piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [S2 if S2 is not None else neg_inf, S1, P, R1,
              R2 if R2 is not None else pos_inf, R3 if R3 is not None else pos_inf]
    if price < levels[0] - buf: return -3
    if price < levels[1] - buf: return -2
    if price < levels[2] - buf: return -1
    if price < levels[3] - buf: return 0
    if R2 is None or price < levels[4] - buf: return +1
    if price < levels[5] - buf: return +2
    return +3

def _three_targets_from_pivots(entry: float, direction: str, piv: dict, step: float):
    ladder = sorted({p for p in [piv.get("S3"),piv.get("S2"),piv.get("S1"),piv["P"],piv["R1"],piv.get("R2"),piv.get("R3")] if p is not None})
    eps = 0.10 * step
    if direction == "BUY":
        ups = [x for x in ladder if x > entry + eps]
        while len(ups) < 3:
            k = len(ups) + 1
            ups.append(entry + (0.7 + 0.7*(k-1))*step)
        return ups[0], ups[1], ups[2]
    else:
        dns = [x for x in ladder if x < entry - eps]
        dns = list(sorted(dns, reverse=True))
        while len(dns) < 3:
            k = len(dns) + 1
            dns.append(entry - (0.7 + 0.7*(k-1))*step)
        return dns[0], dns[1], dns[2]

def _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz_tag, price, atr_val):
    if action not in ("BUY","SHORT"): return tp1,tp2,tp3
    risk = abs(entry - sl); 
    if risk <= 1e-9: return tp1,tp2,tp3
    side = 1 if action == "BUY" else -1
    min_rr   = {"ST":0.80,"MID":1.00,"LT":1.25}
    min_pct  = {"ST":0.006,"MID":0.012,"LT":0.020}
    atr_mult = {"ST":0.50,"MID":0.80,"LT":1.20}
    floor1 = max(min_rr[hz_tag]*risk, min_pct[hz_tag]*price, atr_mult[hz_tag]*atr_val)
    if abs(tp1 - entry) < floor1: tp1 = entry + side*floor1
    floor2 = max(1.6*floor1, (min_rr[hz_tag]*1.8)*risk)
    if abs(tp2 - entry) < floor2: tp2 = entry + side*floor2
    min_gap3 = max(0.8*floor1, 0.6*risk)
    if abs(tp3 - tp2) < min_gap3: tp3 = tp2 + side*min_gap3
    return tp1,tp2,tp3

def _order_targets(entry, tp1, tp2, tp3, action, eps=1e-6):
    side = 1 if action == "BUY" else -1
    arr = sorted([float(tp1),float(tp2),float(tp3)], key=lambda x: side*(x-entry))
    d0 = side*(arr[0]-entry); d1 = side*(arr[1]-entry); d2 = side*(arr[2]-entry)
    if d1 - d0 < eps: arr[1] = entry + side*max(d0 + max(eps,0.1*abs(d0)), d1+eps)
    if side*(arr[2]-entry) - side*(arr[1]-entry) < eps:
        d1 = side*(arr[1]-entry)
        arr[2] = entry + side*max(d1 + max(eps,0.1*abs(d1)), d2+eps)
    return arr[0],arr[1],arr[2]

def _hz_tag(text: str) -> str:
    if "Кратко" in text:  return "ST"
    if "Средне" in text:  return "MID"
    return "LT"

def _max_hold_days(hz_text: str) -> int:
    if "Кратко" in hz_text: return 10
    if "Средне" in hz_text: return 30
    return 90

def _snapshot_decision(df_up_to: pd.DataFrame, horizon_text: str):
    """Правила из strategy, но на 'прошлом' баре для бэктеста (без AI-override)."""
    price = float(df_up_to["close"].iloc[-1])
    cfg = dict(look=60, trend=14, atr=14, use_weekly_atr=False) \
        if "Кратко" in horizon_text else \
        (dict(look=120, trend=28, atr=14, use_weekly_atr=True) if "Средне" in horizon_text else
         dict(look=240, trend=56, atr=14, use_weekly_atr=True))
    hz = _hz_tag(horizon_text)

    tail = df_up_to.tail(cfg["look"])
    rng_low  = float(tail["low"].min())
    rng_high = float(tail["high"].max())
    rng_w    = max(1e-9, rng_high - rng_low)
    pos = (price - rng_low)/rng_w

    closes = df_up_to["close"]
    slope = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)

    atr_d = float(_atr_like(df_up_to, n=cfg["atr"]).iloc[-1])
    atr_w = _weekly_atr(df_up_to) if cfg["use_weekly_atr"] else atr_d
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df_up_to, n=cfg["atr"]*2).iloc[-1]))
    streak    = _streak(closes)

    last_row = df_up_to.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body*1.3 and upper > lower*1.1
    long_lower = lower > body*1.3 and lower > upper*1.1

    hlc = _last_period_hlc(df_up_to, "W-FRI")
    if not hlc:
        hlc = (float(df_up_to["high"].tail(60).max()),
               float(df_up_to["low"].tail(60).min()),
               float(df_up_to["close"].iloc[-1]))
    H,L,C = hlc
    piv = _fib_pivots(H,L,C)
    P,R1,R2 = piv["P"],piv["R1"],piv.get("R2")

    buf = 0.25*atr_w
    band = _classify_band(price, piv, buf)

    last_o, last_c = float(df_up_to["open"].iloc[-1]), float(df_up_to["close"].iloc[-1])
    last_h = float(df_up_to["high"].iloc[-1])
    upper_wick_d = max(0.0, last_h - max(last_o,last_c))
    body_d = abs(last_c - last_o)
    bearish_reject = (last_c < last_o) and (upper_wick_d > body_d)
    very_high_pos = pos >= 0.80

    if very_high_pos:
        if (R2 is not None) and (price > R2 + 0.6*buf) and (slope_norm > 0):
            action, scenario = "BUY","breakout_up"
        else:
            action, scenario = "SHORT","fade_top"
    else:
        if band >= +2:
            action, scenario = ("BUY","breakout_up") if ((R2 is not None) and (price > R2 + 0.6*buf) and (slope_norm > 0)) else ("SHORT","fade_top")
        elif band == +1:
            action, scenario = ("WAIT","upper_wait") if (slope_norm > 0.0015 and not bearish_reject and not long_upper) else ("SHORT","fade_top")
        elif band == 0:
            action, scenario = ("BUY","trend_follow") if slope_norm >= 0 else ("WAIT","mid_range")
        elif band == -1:
            action, scenario = ("BUY","revert_from_bottom") if (streak <= -3 or long_lower) else ("BUY","trend_follow")
        else:
            action, scenario = ("BUY","revert_from_bottom") if band <= -2 else ("WAIT","upper_wait")

    base = 0.55 + 0.12*min(1.0, abs(slope_norm)*1800) + 0.08*min(1.0, max(0.0,(vol_ratio-0.9)/0.6))
    if action == "WAIT": base -= 0.07
    if band >= +1 and action == "BUY": base -= 0.10
    if band <= -1 and action == "BUY": base += 0.05
    conf = float(max(0.55, min(0.90, base)))

    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if scenario == "breakout_up":
            base_ref = R2 if R2 is not None else R1
            entry = max(price, base_ref + 0.10*step_w); sl = base_ref - 1.00*step_w
            tp1,tp2,tp3 = _three_targets_from_pivots(entry,"BUY",piv,step_w)
        elif price < P:
            entry = max(price, piv["S1"] + 0.15*step_w); sl = piv["S1"] - 0.60*step_w
            tp1,tp2,tp3 = _three_targets_from_pivots(entry,"BUY",piv,step_w)
        else:
            entry = max(price, P + 0.10*step_w); sl = P - 0.60*step_w
            tp1,tp2,tp3 = _three_targets_from_pivots(entry,"BUY",piv,step_w)
    elif action == "SHORT":
        if price >= R1:
            entry = min(price, R1 - 0.15*step_w); sl = R1 + 0.60*step_w
        else:
            entry = price + 0.15*step_d; sl = price + 1.00*step_d
        tp1,tp2,tp3 = _three_targets_from_pivots(entry,"SHORT",piv,step_w)
    else:
        entry = price; sl = price - 0.90*step_d
        tp1,tp2,tp3 = entry + 0.7*step_d, entry + 1.4*step_d, entry + 2.1*step_d

    atr_for_floor = atr_w if hz != "ST" else atr_d
    tp1,tp2,tp3 = _apply_tp_floors(entry,sl,tp1,tp2,tp3,action,hz,price,atr_for_floor)
    tp1,tp2,tp3 = _order_targets(entry,tp1,tp2,tp3,action)

    return dict(
        action=action, confidence=conf,
        entry=float(entry), sl=float(sl),
        tp1=float(tp1), tp2=float(tp2), tp3=float(tp3)
    )

def _simulate_outcome(df_all: pd.DataFrame, start_idx: int, action: str,
                      entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                      horizon_text: str,
                      exit_policy: str = "first_touch",   # "first_touch" | "best_tp_before_sl" | "partial_133"
                      tie_rule: str = "SL_first"):         # "SL_first" | "TP_first"
    side = 1 if action == "BUY" else -1
    risk = abs(entry - sl) if entry else 0.0
    max_days = _max_hold_days(horizon_text)

    # для "best_tp_before_sl"
    best_tp_px, best_tp_lab, best_tp_ts = None, None, None

    # для partial
    realized_r = 0.0; realized_ret_pct = 0.0
    realized_px_weight = 0.0; realized_frac = 0.0
    last_fill_ts = df_all.index[start_idx]

    def _apply_fill(px: float, frac: float, ts):
        nonlocal realized_r, realized_ret_pct, realized_px_weight, realized_frac, last_fill_ts
        if frac <= 0: return
        r_part = (side * (px - entry) / risk) * frac if risk > 1e-9 else 0.0
        pct_part = (100.0 * side * (px / entry - 1.0)) * frac if entry else 0.0
        realized_r += r_part; realized_ret_pct += pct_part
        realized_px_weight += px * frac; realized_frac += frac
        last_fill_ts = ts

    for k in range(1, max_days + 1):
        if start_idx + k >= len(df_all): break
        row = df_all.iloc[start_idx + k]
        hi, lo, ts = float(row["high"]), float(row["low"]), row.name

        if action == "BUY":
            sl_hit = lo <= sl
            tps = [("TP3", hi >= tp3, tp3), ("TP2", hi >= tp2, tp2), ("TP1", hi >= tp1, tp1)]
        else:
            sl_hit = hi >= sl
            tps = [("TP3", lo <= tp3, tp3), ("TP2", lo <= tp2, tp2), ("TP1", lo <= tp1, tp1)]

        tps_today = [(lab, px) for (lab, hit, px) in tps if hit]  # уже в порядке TP3→TP1

        if exit_policy == "first_touch":
            if sl_hit and tps_today:
                if tie_rule == "SL_first":
                    exit_px, outcome = sl, "SL"
                else:
                    exit_px, outcome = tps_today[0][1], tps_today[0][0]
            elif sl_hit:
                exit_px, outcome = sl, "SL"
            elif tps_today:
                exit_px, outcome = tps_today[0][1], tps_today[0][0]
            else:
                continue

            r_mult = side * (exit_px - entry) / risk if risk > 1e-9 else 0.0
            ret_pct = 100.0 * side * (exit_px / entry - 1.0) if entry else 0.0
            return dict(exit_ts=ts, exit_price=float(exit_px),
                        outcome=outcome, r_mult=float(r_mult), ret_pct=float(ret_pct),
                        days=int(k))

        elif exit_policy == "best_tp_before_sl":
            if tps_today:
                best_tp_px, best_tp_lab, best_tp_ts = tps_today[0][1], tps_today[0][0], ts
            if sl_hit:
                if tps_today and tie_rule == "TP_first":
                    exit_px, outcome, ex_ts = tps_today[0][1], tps_today[0][0], ts
                elif best_tp_px is not None:
                    exit_px, outcome, ex_ts = best_tp_px, best_tp_lab, best_tp_ts
                else:
                    exit_px, outcome, ex_ts = sl, "SL", ts
                r_mult = side * (exit_px - entry) / risk if risk > 1e-9 else 0.0
                ret_pct = 100.0 * side * (exit_px / entry - 1.0) if entry else 0.0
                return dict(exit_ts=ex_ts, exit_price=float(exit_px),
                            outcome=outcome, r_mult=float(r_mult), ret_pct=float(ret_pct),
                            days=int(k if ex_ts == ts else (ex_ts - df_all.index[start_idx]).days))
            else:
                continue

        else:  # partial_133
            if sl_hit and tps_today and tie_rule == "SL_first":
                _apply_fill(sl, 1.0 - realized_frac, ts)
                avg_px = realized_px_weight / realized_frac if realized_frac > 0 else sl
                outcome = "Partial" if realized_frac < 1.0 else "SL"
                return dict(exit_ts=ts, exit_price=float(avg_px),
                            outcome=outcome, r_mult=float(realized_r), ret_pct=float(realized_ret_pct),
                            days=int(k))

            # сначала TP (дальние → ближние)
            for lab, px in tps_today:
                if realized_frac >= 1.0: break
                _apply_fill(px, min(1.0/3.0, 1.0 - realized_frac), ts)

            if realized_frac >= 1.0:
                avg_px = realized_px_weight / realized_frac
                return dict(exit_ts=ts, exit_price=float(avg_px),
                            outcome="Partial", r_mult=float(realized_r), ret_pct=float(realized_ret_pct),
                            days=int(k))

            if sl_hit:
                _apply_fill(sl, 1.0 - realized_frac, ts)
                avg_px = realized_px_weight / realized_frac if realized_frac > 0 else sl
                outcome = "Partial" if realized_frac < 1.0 else "SL"
                return dict(exit_ts=ts, exit_price=float(avg_px),
                            outcome=outcome, r_mult=float(realized_r), ret_pct=float(realized_ret_pct),
                            days=int(k))

            continue

    # таймаут
    last_row = df_all.iloc[min(start_idx + max_days, len(df_all) - 1)]
    if exit_policy == "partial_133" and realized_frac > 0.0:
        _apply_fill(float(last_row["close"]), 1.0 - realized_frac, last_row.name)
        avg_px = realized_px_weight / realized_frac
        return dict(exit_ts=last_fill_ts, exit_price=float(avg_px),
                    outcome="Partial", r_mult=float(realized_r), ret_pct=float(realized_ret_pct),
                    days=int((last_fill_ts - df_all.index[start_idx]).days))

    exit_px, exit_ts = float(last_row["close"]), last_row.name
    r_mult = side * (exit_px - entry) / risk if risk > 1e-9 else 0.0
    ret_pct = 100.0 * side * (exit_px / entry - 1.0) if entry else 0.0
    return dict(exit_ts=exit_ts, exit_price=float(exit_px),
                outcome="Timeout", r_mult=float(r_mult), ret_pct=float(ret_pct),
                days=int((exit_ts - df_all.index[start_idx]).days))

def scan_signals_no_dupes_with_results(
    ticker_norm: str, horizon_text: str, years: int = 2,
    min_conf: float = 0.70, min_gap_days: int = 7,
    exit_policy: str = "first_touch", tie_rule: str = "SL_first",
):
    cli = PolygonClient()
    days = years * 365 + 300
    df = cli.daily_ohlc(ticker_norm, days=days).copy()
    df = df[~df.index.duplicated(keep="last")]  # на всякий случай
    df = df.sort_index()

    look_start = 260  # позволяем индикаторам прогреться
    signals = []
    last_used_ts = None
    last_action = None

    for i in range(look_start, len(df)-1):
        df_up_to = df.iloc[:i+1]
        snap = _snapshot_decision(df_up_to, horizon_text)
        if snap["action"] == "WAIT": 
            continue
        if snap["confidence"] < float(min_conf):
            continue

        ts = df_up_to.index[-1]

        # анти-дубликаты: разрыв по дням и по направлению
        if last_used_ts is not None:
            if (ts - last_used_ts).days < int(min_gap_days) and snap["action"] == last_action:
                continue

        signals.append(dict(
            ts=ts, action=snap["action"], confidence=float(snap["confidence"]),
            entry=snap["entry"], sl=snap["sl"],
            tp1=snap["tp1"], tp2=snap["tp2"], tp3=snap["tp3"],
            row_i=i
        ))
        last_used_ts = ts
        last_action = snap["action"]

    df_k = pd.DataFrame(signals)
    if df_k.empty:
        return df_k

    res = []
    for r in df_k.itertuples(index=False):
        sim = _simulate_outcome(
            df, r.row_i, r.action, r.entry, r.sl, r.tp1, r.tp2, r.tp3,
            horizon_text, exit_policy=exit_policy, tie_rule=tie_rule
        )
        res.append(sim)
    df_res = pd.DataFrame(res)
    out = pd.concat([df_k.reset_index(drop=True), df_res], axis=1)
    return out

with st.expander("📊 Бэктест по сигналам (без дублей)"):
    bt_ticker = st.text_input("Тикер для бэктеста", value=ticker or "QQQ", key="bt_ticker_input")
    colb1, colb2, colb3 = st.columns(3)
    years = colb1.slider("Годов истории", 1, 5, 2, key="bt_years")
    min_conf = colb2.slider("Мин. уверенность", 0.50, 0.90, 0.62, 0.01, key="bt_min_conf")
    min_gap = colb3.slider("Мин. разрыв между похожими сигналами (дней)", 1, 15, 5, key="bt_mingap")

    colb4, colb5 = st.columns(2)
    with colb4:
        exit_mode_h = st.radio(
            "Правило выхода",
            ["Первое событие (консервативно)", "Лучший TP до SL (оптимистично)", "Частичная фиксация 1/3-1/3-1/3"],
            index=0, key="bt_exit_policy_radio"
        )
        exit_policy = "first_touch" if exit_mode_h.startswith("Первое") else ("best_tp_before_sl" if exit_mode_h.startswith("Лучший") else "partial_133")
    with colb5:
        tie_h = st.radio("Если в одном баре и SL, и TP",
                         ["SL приоритет", "TP приоритет"], index=0, key="bt_tie_rule_radio")
        tie_rule = "SL_first" if tie_h.startswith("SL") else "TP_first"

    if st.button("Запустить отбор сигналов", key="bt_run_btn"):
        try:
            sym_norm = normalize_for_polygon(bt_ticker)
            df_bt = scan_signals_no_dupes_with_results(
                sym_norm, horizon, years, float(min_conf), int(min_gap),
                exit_policy=exit_policy, tie_rule=tie_rule
            )
            st.success(f"Найдено сигналов: {len(df_bt)}")

            avg_r = float(df_bt["r_mult"].mean()) if len(df_bt) else 0.0
            wins = int((df_bt["r_mult"] > 0).sum())
            winrate = (wins / len(df_bt) * 100.0) if len(df_bt) else 0.0
            pf = (df_bt.loc[df_bt["r_mult"] > 0, "r_mult"].sum() /
                  abs(df_bt.loc[df_bt["r_mult"] <= 0, "r_mult"].sum())
                 ) if (len(df_bt) and (df_bt["r_mult"] <= 0).any()) else np.nan

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Сделок", f"{len(df_bt)}")
            c2.metric("Win-rate", f"{winrate:.1f}%")
            c3.metric("Avg R", f"{avg_r:.2f}")
            c4.metric("Profit Factor", "-" if np.isnan(pf) else f"{pf:.2f}")

            show_cols = ["ts","action","confidence","entry","sl","tp1","tp2","tp3",
                         "exit_ts","exit_price","outcome","r_mult","ret_pct","days"]
            for c in show_cols:
                if c not in df_bt.columns: df_bt[c] = np.nan
            st.dataframe(df_bt[show_cols].sort_values("ts").reset_index(drop=True),
                         use_container_width=True)

            csv = df_bt[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Скачать CSV", data=csv,
                               file_name=f"signals_{bt_ticker}_{exit_policy}.csv",
                               mime="text/csv", key="bt_download_csv")

        except Exception as e:
            st.error(f"Ошибка бэктеста: {e}")
