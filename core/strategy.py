import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

# Вспомогательные функции (без раскрытия индикаторов в тексте UI)
def _price_data(ticker: str, horizon: str) -> pd.DataFrame:
    # Выбор периода под горизонт (но пользователю это не показываем)
    if "Кратко" in horizon:
        period = "3mo"
        interval = "1d"
    elif "Средне" in horizon:
        period = "9mo"
        interval = "1d"
    else:
        period = "5y"
        interval = "1d"
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("Нет данных по тикеру.")
    df = df.rename(columns=str.lower)
    return df

def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # Простая прокси-длина волны движения
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _recent_range(df: pd.DataFrame, lookback: int = 60):
    tail = df.tail(lookback)
    return float(tail['low'].min()), float(tail['high'].max())

def _format_price(p: float) -> str:
    # Красиво форматируем числа
    if p is None or np.isnan(p):
        return "—"
    return f"{p:.2f}"

def _build_levels(price: float, direction: str, step: float):
    # direction: LONG / SHORT / WAIT
    if direction == "LONG":
        entry = price - step * 0.2
        sl = price - step * 1.0
        tp1 = price + step * 0.8
        tp2 = price + step * 1.6
    elif direction == "SHORT":
        entry = price + step * 0.2
        sl = price + step * 1.0
        tp1 = price - step * 0.8
        tp2 = price - step * 1.6
    else:  # WAIT — предлагаем реакцию рядом с ценой
        entry = None; sl = None; tp1 = None; tp2 = None
    return {
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "entry_str": _format_price(entry),
        "sl_str": _format_price(sl),
        "tp1_str": _format_price(tp1),
        "tp2_str": _format_price(tp2),
    }

def _direction_from_position(price: float, low: float, high: float):
    # Простейшая логика: середина диапазона — WAIT; ближе к нижней — LONG; ближе к верхней — SHORT
    mid = (low + high) / 2
    width = max(1e-9, high - low)
    pos = (price - low) / width  # 0..1
    if 0.4 < pos < 0.6:
        return "WAIT", "Середина коридора — явного перевеса нет."
    elif pos <= 0.4:
        return "LONG", "Цена ближе к поддержке; следим за реакцией от ближайшей зоны покупателя."
    else:
        return "SHORT", "Цена ближе к сопротивлению; следим за откатом от верхней границы."

def analyze_asset(ticker: str, horizon: str):
    df = _price_data(ticker, horizon)
    price = float(df['close'].iloc[-1])
    low, high = _recent_range(df, lookback=90 if "Долго" in horizon else 60)
    atrp = float(_atr_like(df, n=14).iloc[-1])

    direction, background_hint = _direction_from_position(price, low, high)

    # Шаг движения подбираем от волатильности, но пользователю это не раскрываем
    step = max(1e-6, atrp)
    levels = _build_levels(price, direction if direction in ("LONG","SHORT") else "WAIT", step)

    # Сформируем текстовые блоки
    background = ("Смешанный фон, лучше дождаться реакции рядом с ценой."
                  if direction == "WAIT" else background_hint)

    # Базовый и противоположный сценарий
    if direction == "LONG":
        recommendation = {"action": "BUY"}
        alternative = "Если пойдёт вниз и не удержится зона покупателя — не входить; смотреть повторный сигнал выше и только после подтверждения."
    elif direction == "SHORT":
        recommendation = {"action": "SHORT"}
        alternative = "Если пойдёт выше без отката — не гнаться; ждать возврата и признаки слабости возле недавних максимумов."
    else:
        recommendation = {"action": "WAIT"}
        alternative = "При уверенном выходе из коридора — вход по направлению пробоя после возврата и подтверждения на ретесте."

    comment = background_hint

    return {
        "background": background,
        "recommendation": recommendation,
        "levels": levels,
        "alternative": alternative,
        "comment": comment,
    }
