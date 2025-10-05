from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    # Wilder's smoothing
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = rma(pd.Series(gain, index=close.index), length)
    roll_down = rma(pd.Series(loss, index=close.index), length)
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def atr_wilder(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return rma(tr, length)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    # df must have columns: open, high, low, close
    ha = pd.DataFrame(
        index=df.index,
        columns=["ha_open", "ha_high", "ha_low", "ha_close"],
        dtype=float,
    )
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = []
    for i, (o, h, l, c) in enumerate(df[["open", "high", "low", "close"]].values):
        if i == 0:
            ha_open.append((o + c) / 2.0)
        else:
            prev_ha_o = ha_open[-1]
            prev_ha_c = ha["ha_close"].iloc[i - 1]
            ha_open.append((prev_ha_o + prev_ha_c) / 2.0)
    ha["ha_open"] = ha_open
    ha["ha_high"] = np.stack([df["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(
        axis=1
    )
    ha["ha_low"] = np.stack([df["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(
        axis=1
    )
    return ha


def streak_pos_neg(series: pd.Series, positive: bool = True) -> int:
    # count consecutive bars (including last) that are >=0 (if positive) or <=0 (if not)
    if series.empty:
        return 0
    cnt = 0
    for v in series[::-1]:
        if (v >= 0 and positive) or (v <= 0 and not positive):
            cnt += 1
        else:
            break
    return cnt


def is_hist_slowing(hist: pd.Series, look: int = 5) -> bool:
    # simplistic deceleration: last 3 bars shrink in absolute value vs prior
    if len(hist.dropna()) < 6:
        return False
    last = hist.dropna().iloc[-look:]
    # check two recent decreases in a row in abs value
    dec1 = abs(last.iloc[-3]) >= abs(last.iloc[-2]) and abs(last.iloc[-2]) >= abs(
        last.iloc[-1]
    )
    return bool(dec1)


def recent_divergence(price: pd.Series, rsi: pd.Series, look: int = 60, lag: int = 5):
    # Very simple divergence detector: compare two swing points
    if len(price) < look + lag + 5:
        return None
    p = price.iloc[-(look + lag) :]
    r = rsi.iloc[-(look + lag) :]
    # recent highs
    ih1 = p.idxmax()
    p1 = p.loc[ih1]
    r1 = r.loc[ih1]
    # exclude window around ih1 and find previous high
    p_ex = p.loc[p.index < ih1 - pd.Timedelta(days=1)]
    if p_ex.empty:
        return None
    ih0 = p_ex.idxmax()
    p0 = p.loc[ih0]
    r0 = r.loc[ih0]
    # bearish divergence: higher high in price, lower high in RSI
    bear = p1 > p0 and r.loc[ih1] < r.loc[ih0]
    # bullish divergence: lower low in price, higher low in RSI
    il1 = p.idxmin()
    p1l = p.loc[il1]
    r1l = r.loc[il1]
    p_ex2 = p.loc[p.index < il1 - pd.Timedelta(days=1)]
    if p_ex2.empty:
        bull = False
    else:
        il0 = p_ex2.idxmin()
        p0l = p.loc[il0]
        r0l = r.loc[il0]
        bull = p1l < p0l and r.loc[il1] > r.loc[il0]
    if bear:
        return "bearish"
    if bull:
        return "bullish"
    return None
