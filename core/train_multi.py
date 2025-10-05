import argparse
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

# берем клиента и часть утилит из твоей стратегии
from core.polygon_client import PolygonClient


# ---- helpers (минимально необходимые копии из strategy) ----
def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _linreg_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    beta = ((x - xm) * (y - ym)).sum() / denom
    return float(beta)


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index.copy())
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha["ha_close"].iloc[i - 1]) / 2.0)
    ha["ha_open"] = pd.Series(ha_open, index=df.index)
    return ha


def _streak_by_sign(series: pd.Series, positive: bool = True) -> int:
    sgn = 1 if positive else -1
    run = 0
    vals = series.values
    for i in range(len(vals) - 1, -1, -1):
        v = vals[i]
        if (v > 0 and sgn == 1) or (v < 0 and sgn == -1):
            run += 1
        elif v == 0:
            continue
        else:
            break
    return run


def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(g) < 2:
        return None
    row = g.iloc[-2]
    return float(row["high"]), float(row["low"]), float(row["close"])


def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d
    R2 = P + 0.618 * d
    R3 = P + 1.000 * d
    S1 = P - 0.382 * d
    S2 = P - 0.618 * d
    S3 = P - 1.000 * d
    return {"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3}


def _classify_band(price: float, piv: dict, buf: float) -> int:
    P, R1 = piv["P"], piv["R1"]
    R2, R3, S1, S2 = piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [
        S2 if S2 is not None else neg_inf,
        S1,
        P,
        R1,
        R2 if R2 is not None else pos_inf,
        R3 if R3 is not None else pos_inf,
    ]
    if price < levels[0] - buf:
        return -3
    if price < levels[1] - buf:
        return -2
    if price < levels[2] - buf:
        return -1
    if price < levels[3] - buf:
        return 0
    if R2 is None or price < levels[4] - buf:
        return +1
    if price < levels[5] - buf:
        return +2
    return +3


# ---- feature builder ----
def build_features(df: pd.DataFrame, hz: str) -> pd.DataFrame:
    # базовые окна по горизонту
    look = {"ST": 60, "MID": 120, "LT": 240}[hz]
    trend_win = {"ST": 14, "MID": 28, "LT": 56}[hz]
    atr_n = 14

    out = pd.DataFrame(index=df.index)
    price = df["close"]

    # позиции/наклон
    tail = df["close"].rolling(look)
    rng_low = df["low"].rolling(look).min()
    rng_high = df["high"].rolling(look).max()
    rng_w = (rng_high - rng_low).replace(0, np.nan)
    out["pos"] = (price - rng_low) / rng_w

    out["slope_norm"] = pd.Series(
        [
            (
                _linreg_slope(price.iloc[i - trend_win + 1 : i + 1].values) / max(1e-9, price.iloc[i])
                if i >= trend_win - 1
                else 0.0
            )
            for i in range(len(price))
        ],
        index=df.index,
    )

    atr_d = _atr_like(df, n=atr_n)
    atr_d2 = _atr_like(df, n=atr_n * 2)
    out["atr_d_over_price"] = atr_d / price.replace(0, np.nan)
    out["vol_ratio"] = atr_d / atr_d2.replace(0, np.nan)

    # Heikin Ashi + MACD streaks
    ha = _heikin_ashi(df)
    ha_diff = ha["ha_close"].diff()
    out["ha_up_run"] = pd.Series(
        [_streak_by_sign(ha_diff.iloc[: i + 1], positive=True) for i in range(len(ha_diff))], index=df.index
    )
    out["ha_down_run"] = pd.Series(
        [_streak_by_sign(ha_diff.iloc[: i + 1], positive=False) for i in range(len(ha_diff))], index=df.index
    )

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    out["macd_pos_run"] = pd.Series(
        [_streak_by_sign(hist.iloc[: i + 1], positive=True) for i in range(len(hist))], index=df.index
    )
    out["macd_neg_run"] = pd.Series(
        [_streak_by_sign(hist.iloc[: i + 1], positive=False) for i in range(len(hist))], index=df.index
    )

    # pivots & band
    rule = "W-FRI" if hz == "ST" else "M"
    bands = []
    buf_mult = {"ST": 0.18, "MID": 0.22, "LT": 0.28}[hz]
    # сделаем дневной ATR как масштаб буфера
    atr_last = atr_d.fillna(method="ffill")
    for i in range(len(df)):
        part = df.iloc[: i + 1]
        hlc = _last_period_hlc(part, rule)
        if not hlc:
            bands.append(0)
            continue
        H, L, C = hlc
        piv = _fib_pivots(H, L, C)
        pr = float(df["close"].iloc[i])
        buf = buf_mult * float(atr_last.iloc[i] if not np.isnan(atr_last.iloc[i]) else 0.0)
        bands.append(_classify_band(pr, piv, buf))
    out["band"] = pd.Series(bands, index=df.index)

    return out


def make_labels(df: pd.DataFrame, hz: str) -> pd.Series:
    horizon_days = {"ST": 5, "MID": 20, "LT": 60}[hz]
    future = df["close"].shift(-horizon_days)
    ret = (future - df["close"]) / df["close"]
    # метка: 1 если через N дней выше, иначе 0
    y = (ret > 0).astype(int)
    return y


# ---- main training ----
def train_multi(hz: str, tickers: list[str], years: int, out_path: str | None):
    cli = PolygonClient()

    frames = []
    ys = []
    for t in tickers:
        # запас дней к истории
        days = int(years * 365 + 120)
        print(f"[{t}] fetching {days} days...")
        df = cli.daily_ohlc(t, days=days).dropna()
        if df.empty or len(df) < 200:
            print(f"[{t}] skipped: not enough data")
            continue

        X = build_features(df, hz)
        y = make_labels(df, hz)

        # синхронизация индексов/NaN
        data = pd.concat([X, y.rename("y")], axis=1).dropna()
        if data.empty:
            print(f"[{t}] no rows after dropna, skip")
            continue

        data["ticker"] = t  # как фича тоже
        frames.append(data.drop(columns=["y"]))
        ys.append(data["y"])

    if not frames:
        raise RuntimeError("Dataset is empty — проверь тикеры/годы/Polygon API.")

    X_all = pd.concat(frames, axis=0)
    y_all = pd.concat(ys, axis=0)

    # простая обработка категориальной "ticker"
    X_all = pd.get_dummies(X_all, columns=["ticker"], drop_first=False)

    # train/val split по времени (последние 15% на валидацию)
    split_idx = int(len(X_all) * 0.85)
    X_tr, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_tr, y_val = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

    # модель без внешних зависимостей: HistGradientBoostingClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, roc_auc_score

    clf = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=None,
        max_iter=400,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=42,
    )
    print("Training...")
    clf.fit(X_tr, y_tr)

    # метрики
    p_val = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val)
    print(f"VAL AUC: {auc:.3f}")
    print(classification_report(y_val, (p_val >= 0.5).astype(int)))

    # --- сохраняем ---
    model_pack = {
        "model": clf,
        "features": list(X_all.columns),
        "hz": hz,
        "tickers": tickers,
        "meta": {"tickers": tickers},
        "notes": "multi-ticker model; label = future N-day return > 0",
    }

    if out_path is None:
        out_path = os.path.join("models", f"arxora_lgbm_{hz}.joblib")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model_pack, out_path)
    print(f"Saved model to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", required=True, choices=["ST", "MID", "LT"], help="horizon")
    ap.add_argument("--tickers", required=True, help="comma-separated tickers")
    ap.add_argument("--years", type=int, default=6, help="years of history")
    ap.add_argument("--out", default=None, help="optional output path")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    train_multi(args.hz, tickers, args.years, args.out)


if __name__ == "__main__":
    main()
