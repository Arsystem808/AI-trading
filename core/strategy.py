from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from .utils import fib_pivots, last_consecutive_run, pct_diff
from .indicators import heikin_ashi, macd_hist, rsi_wilder, atr_wilder, streak_pos_neg, is_hist_slowing, recent_divergence

@dataclass
class HorizonParams:
    name: str
    period: str  # 'W'/'M'/'Y' for pivots
    ha_series_min: int
    macd_streak_min: int
    tol: float   # proximity tolerance for R2/S2 checks
    atr_stop_mult: float
    atr_tp1_mult: float
    atr_tp2_mult: float

HORIZONS: Dict[str, HorizonParams] = {
    "ST": HorizonParams("Краткосрок (1–5 дней)", "W", 4, 4, 0.008, 0.8, 0.6, 1.1),
    "MID": HorizonParams("Среднесрок (1–4 недели)", "M", 5, 6, 0.010, 1.0, 0.8, 1.3),
    "LT": HorizonParams("Долгосрок (1–6 месяцев)", "Y", 6, 8, 0.012, 1.3, 1.5, 2.4),
}

def analyze(symbol: str, df_daily: pd.DataFrame, horizon_key: str = "ST") -> Dict[str, Any]:
    hp = HORIZONS[horizon_key]
    # Heikin Ashi & metrics on DAILY bars
    ha = heikin_ashi(df_daily)
    ha_green = (ha['ha_close'] >= ha['ha_open'])
    ha_run = last_consecutive_run(ha_green)  # length of current color run
    # MACD/RSI/ATR
    macd_line, signal_line, hist = macd_hist(df_daily['close'])
    rsi = rsi_wilder(df_daily['close'])
    atr = atr_wilder(df_daily)
    last_close = float(df_daily['close'].iloc[-1])
    last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else np.nan
    
    # Streaks
    pos_streak = streak_pos_neg(hist, positive=True)
    neg_streak = streak_pos_neg(hist, positive=False)
    slowing = is_hist_slowing(hist)
    div = recent_divergence(df_daily['close'], rsi)
    
    # Pivots from previous period (weekly/monthly/yearly)
    from .data import prev_period_hlc_from_daily
    ph, pl, pc = prev_period_hlc_from_daily(df_daily, period=hp.period)
    piv = fib_pivots(ph, pl, pc)
    
    # Context checks
    near_R2 = (last_close >= piv.R2) or (pct_diff(last_close, piv.R2) <= hp.tol)
    near_S2 = (last_close <= piv.S2) or (pct_diff(last_close, piv.S2) <= hp.tol)
    near_R3 = (last_close >= piv.R3) or (pct_diff(last_close, piv.R3) <= hp.tol)
    near_S3 = (last_close <= piv.S3) or (pct_diff(last_close, piv.S3) <= hp.tol)
    
    green_series = ha_green.iloc[-1] and ha_run >= hp.ha_series_min
    red_series   = (not ha_green.iloc[-1]) and ha_run >= hp.ha_series_min
    macd_green_ok = hist.iloc[-1] >= 0 and pos_streak >= hp.macd_streak_min
    macd_red_ok   = hist.iloc[-1] <= 0 and neg_streak >= hp.macd_streak_min
    
    confidence = 0.50
    # Add signals to confidence
    if green_series or red_series: confidence += 0.10
    if macd_green_ok or macd_red_ok: confidence += 0.10
    if slowing: confidence += 0.05
    if div == 'bearish' and near_R2: confidence += 0.10
    if div == 'bullish' and near_S2: confidence += 0.10
    confidence = float(np.clip(confidence, 0.30, 0.90))
    
    # Decision logic
    action = "WAIT"; entry=None; tp1=None; tp2=None; sl=None; alt_action="WAIT"; alt_note=""; commentary=""
    
    # A) Overheat near roof: default WAIT, alt SHORT
    if near_R2 and green_series and (macd_green_ok or slowing):
        commentary = "У верхней границы после затяжного роста — импульс выдыхается, логична перезагрузка к середине диапазона."
        action = "WAIT"
        alt_action = "SHORT (агрессивно)"
        # Entry at current / a bit higher
        entry = last_close
        # Stop logic: behind R3 if price very high, else behind R2
        if near_R3:
            sl = piv.R3 + hp.atr_stop_mult * last_atr
            tp1 = piv.R2
            tp2 = piv.P
        else:
            sl = piv.R2 + hp.atr_stop_mult * last_atr
            tp1 = (piv.P + piv.S1) / 2.0
            tp2 = piv.S1
        alt_note = "Вход от текущих/чуть выше; риски повышены — нужен аккуратный размер позиции."
    
    # B) Oversold near floor: base LONG, alt WAIT
    elif near_S2 and red_series and (macd_red_ok or slowing):
        commentary = "У нижней границы после затяжного снижения видны признаки выдохшегося импульса — разумно ждать откат вверх."
        action = "LONG"
        entry = last_close
        if near_S3:
            sl = piv.S3 - hp.atr_stop_mult * last_atr
            tp1 = piv.S2
            tp2 = piv.P
        else:
            sl = piv.S2 - hp.atr_stop_mult * last_atr
            tp1 = (piv.P + piv.R1) / 2.0
            tp2 = piv.R1
        alt_action = "WAIT"
        alt_note = "Если импульс вниз ещё не исчерпан — лучше дождаться признаков остановки."
    
    else:
        # Middle zone → WAIT with conditional plans
        commentary = "Цена в середине коридора — явного преимущества нет, разумнее подождать реакцию у сильных зон."
        action = "WAIT"
        alt_action = "LONG/SHORT от границ"
        alt_note = "План: работать от крайних зон — при слабости сверху искать вход вниз; у основания — искать разворот вверх."
        # propose conditional stop/targets with ATR for orientation
        entry = None
        tp1 = None
        tp2 = None
        sl = None
    
    # Ensure logical prices
    def clean(x):
        return None if (x is None or np.isnan(x)) else float(x)
    entry, tp1, tp2, sl = map(clean, [entry, tp1, tp2, sl])
    
    # Bounds sanity: avoid stop above entry for LONG etc.
    if action.startswith("LONG") and entry and sl and sl > entry:
        sl = entry - 0.8 * last_atr
    if action.startswith("SHORT") and entry and sl and sl < entry:
        sl = entry + 0.8 * last_atr
    
    # Pack debug (developer-only)
    debug = {
        "last_close": last_close,
        "last_atr": last_atr,
        "ha_run": ha_run,
        "hist_last": float(hist.iloc[-1]),
        "macd_pos_streak": pos_streak,
        "macd_neg_streak": neg_streak,
        "slowing": slowing,
        "divergence": div,
        "pivots": {
            "P": piv.P, "R1": piv.R1, "R2": piv.R2, "R3": piv.R3,
            "S1": piv.S1, "S2": piv.S2, "S3": piv.S3
        },
        "near": {"R2": near_R2, "S2": near_S2, "R3": near_R3, "S3": near_S3},
        "horizon": hp.name
    }
    
    return {
        "action": action,
        "entry": entry,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "alt_action": alt_action,
        "alt_note": alt_note,
        "commentary": commentary,
        "confidence": confidence,
        "debug": debug
    }
