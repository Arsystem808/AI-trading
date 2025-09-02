import numpy as np
from dataclasses import dataclass
from core.polygon_client import PolygonClient

@dataclass
class Levels:
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float

def _atr_like(df, n=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = np.max([hl, hc, lc], axis=0)
    import pandas as pd
    return pd.Series(tr, index=df.index).rolling(n, min_periods=1).mean()

def _position_bias(price, low, high):
    mid = (low + high)/2
    width = max(1e-9, high - low)
    pos = (price - low) / width  # 0..1
    if 0.45 < pos < 0.55:
        return "WAIT", 0.5, "Середина коридора — явного перевеса нет."
    if pos <= 0.45:
        # ближе к поддержке
        conf = 0.65 - (pos*0.3)  # ниже — чуть увереннее LONG
        return "BUY", float(np.clip(conf, 0.55, 0.9)), "Цена ближе к поддержке; следим за реакцией возле нижних уровней."
    # ближе к сопротивлению
    conf = 0.65 - ((1-pos)*0.3)
    return "SHORT", float(np.clip(conf, 0.55, 0.9)), "Цена ближе к сопротивлению; смотрим на слабость у верхней границы."

def _horizon_days(text):
    if "Кратко" in text: return 20
    if "Средне" in text: return 120
    return 720

def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    days = _horizon_days(horizon)
    df = cli.daily_ohlc(ticker, days=max(90, days))
    price = cli.last_trade_price(ticker)

    # недавний коридор
    look = 60 if days < 180 else 90
    low = float(df['low'].tail(look).min())
    high = float(df['high'].tail(look).max())

    # волатильность
    atrp = float(_atr_like(df, n=14).iloc[-1])
    step = max(1e-6, atrp)

    action, conf, note = _position_bias(price, low, high)

    # уровни
    if action == "BUY":
        entry = price - step*0.15
        sl = price - step*1.0
        tp1 = price + step*0.8
        tp2 = price + step*1.6
        tp3 = price + step*2.4
        alt = "Если уйдёт ниже зоны покупателя — пропустить вход и ждать возврата с подтверждением сверху."
    elif action == "SHORT":
        entry = price + step*0.15
        sl = price + step*1.0
        tp1 = price - step*0.8
        tp2 = price - step*1.6
        tp3 = price - step*2.4
        alt = "Если пробьёт верх и удержится — не гнаться; ждать возврата и признаки слабости у максимумов."
    else:
        # WAIT: предложим пробойный сценарий
        conf = 0.5
        entry = price
        sl = price - step*0.9
        tp1 = price + step*0.7
        tp2 = price + step*1.4
        tp3 = price + step*2.1
        alt = "При уверенном пробое диапазона работаем по направлению после ретеста и подтверждения."

    # условные вероятности достижения целей (эвристика)
    probs = {"tp1": 0.75, "tp2": 0.55, "tp3": 0.28}

    return {
        "last_price": float(price),
        "recommendation": {"action": "BUY" if action=="BUY" else ("SHORT" if action=="SHORT" else "WAIT"), "confidence": float(conf)},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": probs,
        "note": note,
        "alt": alt,
    }
