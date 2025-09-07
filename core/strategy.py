\
"""
Arxora core strategy (public interface).

Outputs: action (BUY/SHORT/CLOSE/WAIT), entry, tp1, tp2, sl, confidence, comment.
Internal logic uses price behavior and levels; UI should not reveal indicators.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class Decision:
    action: str
    entry: str
    tp1: str
    tp2: str
    sl: str
    confidence: int
    comment: str

# ---------- Helpers
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def fib_pivots_from_hlc(h: float, l: float, c: float):
    p = (h + l + c) / 3.0
    rng = (h - l)
    r1 = p + 0.382 * rng
    r2 = p + 0.618 * rng
    r3 = p + 1.000 * rng
    s1 = p - 0.382 * rng
    s2 = p - 0.618 * rng
    s3 = p - 1.000 * rng
    return {"P": p, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}

def timeframe_params(horizon: str):
    horizon = (horizon or "ST").upper()
    if horizon == "ST":
        return dict(pivot_rule="daily_prev", tol=0.004)  # 0.4%
    if horizon == "MID":
        return dict(pivot_rule="weekly_prev", tol=0.006)  # 0.6%
    if horizon == "LT":
        return dict(pivot_rule="monthly_prev", tol=0.008)  # 0.8%
    return dict(pivot_rule="daily_prev", tol=0.004)

def previous_period_hlc(df: pd.DataFrame, rule: str):
    # df must be daily OHLC at minimum
    if rule == "daily_prev":
        d = df.iloc[-2]  # yesterday
        return float(d["High"]), float(d["Low"]), float(d["Close"])
    if rule == "weekly_prev":
        wk = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
        d = wk.iloc[-2]
        return float(d["High"]), float(d["Low"]), float(d["Close"])
    if rule == "monthly_prev":
        mo = df.resample("M").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
        d = mo.iloc[-2]
        return float(d["High"]), float(d["Low"]), float(d["Close"])
    d = df.iloc[-2]
    return float(d["High"]), float(d["Low"]), float(d["Close"])

def streak(series: pd.Series) -> int:
    """Length of current run of same sign for MACD histogram."""
    s = np.sign(series.dropna())
    if s.empty:
        return 0
    run = 0
    last = s.iloc[-1]
    for v in reversed(s.tolist()):
        if v == last:
            run += 1
        else:
            break
    return run

def simple_divergence(price: pd.Series, osc: pd.Series, lookback: int = 80):
    """Return 'bearish'/'bullish'/None approximate divergence label."""
    p = price.tail(lookback)
    o = osc.reindex_like(price).tail(lookback)
    # highs
    hi_idx = p.idxmax()
    p_1h = p.loc[hi_idx]
    # previous high excluding last 10 bars around max
    prev = p.loc[p.index < hi_idx].tail(lookback)
    if len(prev) > 10:
        hi2_idx = prev.idxmax()
        p_2h = prev.loc[hi2_idx]
        o1 = float(o.loc[hi_idx])
        o2 = float(o.loc[hi2_idx])
        if p_1h > p_2h * 0.999 and o1 < o2:  # price higher high, osc lower high
            return "bearish"
    # lows
    lo_idx = p.idxmin()
    p_1l = p.loc[lo_idx]
    prevl = p.loc[p.index < lo_idx].tail(lookback)
    if len(prevl) > 10:
        lo2_idx = prevl.idxmin()
        p_2l = prevl.loc[lo2_idx]
        o1 = float(o.loc[lo_idx])
        o2 = float(o.loc[lo2_idx])
        if p_1l < p_2l * 1.001 and o1 > o2:  # price lower low, osc higher low
            return "bullish"
    return None

def round_lvl(x, tick=0.01):
    if x is None or x == "—":
        return "—"
    return f"{np.round(x / tick) * tick:.2f}"

# ---------- Core decision
def decide(df: pd.DataFrame, horizon: str = "ST") -> Decision:
    """
    df: OHLC with DateTimeIndex, columns ['Open','High','Low','Close']
    horizon: 'ST' | 'MID' | 'LT'
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    params = timeframe_params(horizon)
    h, l, c_prev = previous_period_hlc(df[["High","Low","Close"]], params["pivot_rule"])
    piv = fib_pivots_from_hlc(h, l, c_prev)
    close = float(df["Close"].iloc[-1])

    macd_line, signal_line, hist = macd(df["Close"])
    rsi_val = rsi(df["Close"]).iloc[-1]
    hist_streak = streak(hist)
    turning_down = hist.iloc[-1] < hist.iloc[-2] if len(hist) >= 2 else False
    turning_up = hist.iloc[-1] > hist.iloc[-2] if len(hist) >= 2 else False
    div = simple_divergence(df["Close"], rsi(df["Close"]))

    # Proximity to levels
    tol = params["tol"]
    near = lambda lvl: abs(close - lvl) / max(1e-9, lvl) <= tol

    action = "WAIT"
    entry = tp1 = tp2 = sl = "—"
    confidence = 55
    comment = "Ситуация неоднозначная — ждём реакции цены у ключевой зоны."

    # Bearish setup near resistance
    if near(piv["R1"]) or near(piv["R2"]):
        bear_score = 0
        if turning_down: bear_score += 1
        if hist.iloc[-1] > 0 and hist_streak >= 4: bear_score += 1
        if div == "bearish": bear_score += 1
        if bear_score >= 1:
            action = "SHORT"
            entry = close
            tp1 = (piv["P"] + close)/2 if close > piv["P"] else piv["P"]
            tp2 = piv["S1"]
            sl = piv["R2"] + (piv["R2"]-piv["R1"])*0.25
            confidence = 60 + 10*min(bear_score,3)
            comment = "Цена у области предложения; импульс теряет силу — приоритет аккуратного входа вниз."
    # Bullish setup near support
    if action == "WAIT" and (near(piv["S1"]) or near(piv["S2"])):
        bull_score = 0
        if turning_up: bull_score += 1
        if hist.iloc[-1] < 0 and hist_streak >= 4: bull_score += 1
        if div == "bullish": bull_score += 1
        if bull_score >= 1:
            action = "LONG"
            entry = close
            tp1 = (piv["P"] + close)/2 if close < piv["P"] else piv["P"]
            tp2 = piv["R1"]
            sl = piv["S2"] - (piv["S1"]-piv["S2"])*0.25
            confidence = 60 + 10*min(bull_score,3)
            comment = "Цена у области спроса; появляется упругость движения — ставим приоритет на рост."

    # If far from levels — WAIT
    # Round levels nicely
    tick = _infer_tick(df["Close"])
    entry = round_lvl(entry, tick)
    tp1 = round_lvl(tp1, tick)
    tp2 = round_lvl(tp2, tick)
    sl = round_lvl(sl, tick)

    return Decision(action, entry, tp1, tp2, sl, int(confidence), comment)

def _infer_tick(series: pd.Series) -> float:
    # rough guess based on magnitude
    last = float(series.iloc[-1])
    if last >= 1000: return 1.0
    if last >= 100: return 0.1
    if last >= 10: return 0.01
    if last >= 1: return 0.001
    return 0.0001
