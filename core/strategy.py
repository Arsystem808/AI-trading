
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np

from .utils import fib_pivots, last_consecutive_run, pct_diff, clamp
from .indicators import (
    heikin_ashi, macd_hist, rsi_wilder, atr_wilder,
    streak_pos_neg, is_hist_slowing, recent_divergence
)
from .data import prev_period_hlc_from_daily

@dataclass
class HorizonParams:
    name: str
    period: str  # 'W'/'M'/'Y'
    ha_series_min: int
    macd_streak_min: int
    tol: float
    atr_stop_mult: float
    atr_tp1_mult: float
    atr_tp2_mult: float
    alt_dist_cap: float  # допустимая удалённость альтернативного входа в ATR

HORIZONS: Dict[str, HorizonParams] = {
    "ST":  HorizonParams("Краткосрок (1–5 дней)",      "W", 4, 4, 0.008, 0.8, 0.6, 1.1, 1.2),
    "MID": HorizonParams("Среднесрок (1–4 недели)",    "M", 5, 6, 0.010, 1.0, 0.8, 1.3, 1.8),
    "LT":  HorizonParams("Долгосрок (1–6 месяцев)",    "Y", 6, 8, 0.012, 1.3, 1.2, 2.0, 2.5),
}

def _mid(a: float, b: float) -> float:
    return (a + b) / 2.0

def _sanitize_targets(action: str, entry, tp1, tp2, atr, up_hint, dn_hint, hp) -> tuple:
    """
    Гарантирует корректное направление целей относительно entry.
    up_hint/dn_hint — ориентиры «вверх»/«вниз» (например, середины диапазонов).
    """
    if entry is None:
        return entry, tp1, tp2
    if action.startswith("LONG"):
        if tp1 is None: tp1 = entry + hp.atr_tp1_mult * atr
        if tp2 is None: tp2 = entry + hp.atr_tp2_mult * atr
        if tp1 <= entry: tp1 = max(entry + hp.atr_tp1_mult * atr, up_hint)
        if tp2 <= tp1:   tp2 = max(tp1 + 0.7 * atr, up_hint + 0.5 * atr)
    elif action.startswith("SHORT"):
        if tp1 is None: tp1 = entry - hp.atr_tp1_mult * atr
        if tp2 is None: tp2 = entry - hp.atr_tp2_mult * atr
        if tp1 >= entry: tp1 = min(entry - hp.atr_tp1_mult * atr, dn_hint)
        if tp2 >= tp1:   tp2 = min(tp1 - 0.7 * atr, dn_hint - 0.5 * atr)
    return entry, tp1, tp2

def _fix_stop_for_direction(action: str, entry, sl, atr) -> float:
    if entry is None or sl is None:
        return sl
    if action.startswith("LONG") and sl >= entry:
        return entry - 0.8 * atr
    if action.startswith("SHORT") and sl <= entry:
        return entry + 0.8 * atr
    return sl

def _alt_is_too_far(alt_entry: float, last_close: float, atr: float, cap_atr: float) -> bool:
    if alt_entry is None or atr <= 0:
        return True
    return abs(alt_entry - last_close) > cap_atr * atr

def analyze(symbol: str, df_daily: pd.DataFrame, horizon_key: str = "ST") -> Dict[str, Any]:
    hp = HORIZONS[horizon_key]

    # === Series & indicators ===
    ha = heikin_ashi(df_daily)
    ha_green = (ha['ha_close'] >= ha['ha_open'])
    ha_run = last_consecutive_run(ha_green)
    ha_turn_down = (ha_green.iloc[-2] if len(ha_green) > 1 else True) and (not ha_green.iloc[-1])
    ha_turn_up   = ((not ha_green.iloc[-2]) if len(ha_green) > 1 else False) and ha_green.iloc[-1]

    macd_line, signal_line, hist = macd_hist(df_daily['close'])
    rsi = rsi_wilder(df_daily['close'])
    atr = atr_wilder(df_daily)
    last_close = float(df_daily['close'].iloc[-1])
    last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else max(1e-6, float(df_daily['close'].tail(14).std()))

    pos_streak = streak_pos_neg(hist, positive=True)
    neg_streak = streak_pos_neg(hist, positive=False)
    slowing = is_hist_slowing(hist)
    div = recent_divergence(df_daily['close'], rsi)

    # === Pivots из предыдущего периода (цены в вычислениях, но в тексте не упоминаем уровни) ===
    ph, pl, pc = prev_period_hlc_from_daily(df_daily, period=hp.period)
    piv = fib_pivots(ph, pl, pc)

    # Близость к опорным зонам (без упоминаний в тексте)
    near_R2 = (last_close >= piv.R2) or (pct_diff(last_close, piv.R2) <= hp.tol)
    near_S2 = (last_close <= piv.S2) or (pct_diff(last_close, piv.S2) <= hp.tol)
    near_R3 = (last_close >= piv.R3) or (pct_diff(last_close, piv.R3) <= hp.tol)
    near_S3 = (last_close <= piv.S3) or (pct_diff(last_close, piv.S3) <= hp.tol)

    # Ориентиры «середины» (для целей и хелперов)
    up_hint = _mid(piv.P, piv.R1)
    dn_hint = _mid(piv.P, piv.S1)

    green_series = ha_green.iloc[-1] and ha_run >= hp.ha_series_min
    red_series   = (not ha_green.iloc[-1]) and ha_run >= hp.ha_series_min
    macd_green_ok = hist.iloc[-1] >= 0 and pos_streak >= hp.macd_streak_min
    macd_red_ok   = hist.iloc[-1] <= 0 and neg_streak >= hp.macd_streak_min

    # === Confidence ===
    confidence = 0.50
    if green_series or red_series: confidence += 0.10
    if macd_green_ok or macd_red_ok: confidence += 0.10
    if slowing: confidence += 0.05
    if div == 'bearish' and near_R2: confidence += 0.10
    if div == 'bullish' and near_S2: confidence += 0.10
    if (near_R2 or near_R3) and ha_turn_down: confidence += 0.08
    if (near_S2 or near_S3) and ha_turn_up:   confidence += 0.08
    confidence = clamp(confidence, 0.35, 0.90)

    # === Outputs ===
    action = "WAIT"
    entry = tp1 = tp2 = sl = None
    alt_action = "WAIT"
    alt_entry = alt_tp1 = alt_tp2 = alt_sl = None
    alt_note = ""
    commentary = ""

    # === Верхняя зона (перегрев) ===
    if (near_R3 or near_R2) and (green_series or macd_green_ok or slowing):
        if ha_turn_down or (div == 'bearish') or (slowing and not ha_green.iloc[-1]):
            action = "SHORT"
            entry = last_close
            # стоп за «потолком» + ATR
            if near_R3:
                sl = piv.R3 + hp.atr_stop_mult * last_atr
                tp1 = piv.R2
                tp2 = piv.P
            else:
                sl = piv.R2 + hp.atr_stop_mult * last_atr
                tp1 = dn_hint
                tp2 = piv.S1
            commentary = "Сверху видны признаки усталости — берём откат вниз."
            alt_action = "WAIT"
            alt_note = "Если случится резкий укол вверх — лучше переждать и переоценить."
        else:
            action = "WAIT"
            commentary = "Наверху импульс натянут, но чётких подтверждений разворота пока мало."
            # Альтернатива: агрессивный SHORT «от отката сверху»
            alt_action = "SHORT (агрессивно, от отката сверху)"
            # Альтернативный вход опираем на текущую цену, не уводим слишком далеко
            if near_R3:
                alt_entry = max(last_close, piv.R3 - 0.2 * last_atr)
                alt_sl    = (piv.R3 + hp.atr_stop_mult * last_atr)
                alt_tp1   = piv.R2
                alt_tp2   = piv.P
            else:
                alt_entry = max(last_close, piv.R2 - 0.2 * last_atr)
                alt_sl    = (piv.R2 + hp.atr_stop_mult * last_atr)
                alt_tp1   = dn_hint
                alt_tp2   = piv.S1
            # Если слишком далеко от текущей — убираем альтернативу
            if _alt_is_too_far(alt_entry, last_close, last_atr, hp.alt_dist_cap):
                alt_action = "WAIT"
                alt_entry = alt_tp1 = alt_tp2 = alt_sl = None
                alt_note = "Альтернативный вход слишком далеко от цены — лучше дождаться реакции ближе."
            else:
                alt_note = "Размер позиции умеренный; работаем от отката сверху, цены указаны."
    # === Нижняя зона (перепроданность) ===
    elif (near_S3 or near_S2) and (red_series or macd_red_ok or slowing):
        if ha_turn_up or (div == 'bullish') or (slowing and ha_green.iloc[-1]):
            action = "LONG"
            entry = last_close
            if near_S3:
                sl = piv.S3 - hp.atr_stop_mult * last_atr
                tp1 = piv.S2
                tp2 = piv.P
            else:
                sl = piv.S2 - hp.atr_stop_mult * last_atr
                tp1 = up_hint
                tp2 = piv.R1
            commentary = "Снизу признаки выдохшегося снижения — берём восстановление вверх."
            alt_action = "WAIT"
            alt_note = "Если будет ещё один укол вниз — дождаться остановки."
        else:
            action = "WAIT"
            commentary = "Внизу намёки на разворот есть, но подтверждений мало."
            alt_action = "LONG (аккуратно, от отката снизу)"
            if near_S3:
                alt_entry = min(last_close, piv.S3 + 0.2 * last_atr)
                alt_sl    = (piv.S3 - hp.atr_stop_mult * last_atr)
                alt_tp1   = piv.S2
                alt_tp2   = piv.P
            else:
                alt_entry = min(last_close, piv.S2 + 0.2 * last_atr)
                alt_sl    = (piv.S2 - hp.atr_stop_mult * last_atr)
                alt_tp1   = up_hint
                alt_tp2   = piv.R1
            if _alt_is_too_far(alt_entry, last_close, last_atr, hp.alt_dist_cap):
                alt_action = "WAIT"
                alt_entry = alt_tp1 = alt_tp2 = alt_sl = None
                alt_note = "Альтернативный вход слишком далеко — ждём цены ближе."
            else:
                alt_note = "Входим от отката снизу небольшим объёмом; цены указаны."
    # === Середина ===
    else:
        tilt_up = (pos_streak > neg_streak and ha_green.iloc[-1])
        tilt_dn = (neg_streak > pos_streak and not ha_green.iloc[-1])
        if tilt_up:
            action = "LONG (консервативно)"
            entry = last_close
            sl = entry - hp.atr_stop_mult * last_atr
            tp1 = up_hint
            tp2 = piv.R1
            commentary = "Перевес небольшой, но спрос выглядит сильнее — берём аккуратно с близкими целями."
            alt_action = "SHORT (от отката сверху)"
            alt_entry  = last_close + 0.3 * last_atr
            alt_sl     = alt_entry + hp.atr_stop_mult * last_atr
            alt_tp1    = _mid(piv.P, piv.S1)
            alt_tp2    = piv.S1
        elif tilt_dn:
            action = "SHORT (консервативно)"
            entry = last_close
            sl = entry + hp.atr_stop_mult * last_atr
            tp1 = dn_hint
            tp2 = piv.S1
            commentary = "Небольшой перевес за продавцами — работаем аккуратно."
            alt_action = "LONG (от отката снизу)"
            alt_entry  = last_close - 0.3 * last_atr
            alt_sl     = alt_entry - hp.atr_stop_mult * last_atr
            alt_tp1    = _mid(piv.P, piv.R1)
            alt_tp2    = piv.R1
        else:
            action = "WAIT"
            commentary = "Середина коридора — явного перевеса нет. Ждём реакции рядом с ценой."
            alt_action = "Триггерный вход при выходе из коридора"
            alt_note = "Сценарий: при движении вверх/вниз к ближайшей границе — смотреть на откат и реакцию; входить только после подтверждения."
            alt_entry = alt_tp1 = alt_tp2 = alt_sl = None

    # === Direction sanity ===
    entry, tp1, tp2 = _sanitize_targets(action, entry, tp1, tp2, last_atr, up_hint, dn_hint, hp)
    sl = _fix_stop_for_direction(action, entry, sl, last_atr)
    if alt_action.startswith("LONG"):
        alt_entry, alt_tp1, alt_tp2 = _sanitize_targets("LONG", alt_entry, alt_tp1, alt_tp2, last_atr, up_hint, dn_hint, hp)
        alt_sl = _fix_stop_for_direction("LONG", alt_entry if alt_entry else last_close, alt_sl, last_atr)
    if alt_action.startswith("SHORT"):
        alt_entry, alt_tp1, alt_tp2 = _sanitize_targets("SHORT", alt_entry, alt_tp1, alt_tp2, last_atr, up_hint, dn_hint, hp)
        alt_sl = _fix_stop_for_direction("SHORT", alt_entry if alt_entry else last_close, alt_sl, last_atr)

    debug = {
        "last_close": last_close,
        "last_atr": last_atr,
        "ha_run": ha_run,
        "ha_turn_down": ha_turn_down,
        "ha_turn_up": ha_turn_up,
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
        "entry": None if entry is None else float(entry),
        "tp1": None if tp1 is None else float(tp1),
        "tp2": None if tp2 is None else float(tp2),
        "sl": None if sl is None else float(sl),
        "alt_action": alt_action,
        "alt_entry": None if alt_entry is None else float(alt_entry),
        "alt_tp1": None if alt_tp1 is None else float(alt_tp1),
        "alt_tp2": None if alt_tp2 is None else float(alt_tp2),
        "alt_sl": None if alt_sl is None else float(alt_sl),
        "alt_note": alt_note,
        "commentary": commentary,
        "confidence": float(confidence),
        "debug": debug
    }
