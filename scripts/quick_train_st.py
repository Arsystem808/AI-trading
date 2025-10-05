# scripts/quick_train_st.py
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# берём твой клиент и функции прямо из стратегии
from core.polygon_client import PolygonClient
from core.strategy import (
    _apply_tp_floors,
    _atr_like,
    _classify_band,
    _fib_pivots,
    _horizon_cfg,
    _hz_tag,
    _last_period_hlc,
    _linreg_slope,
    _order_targets,
    _streak,
    _three_targets_from_pivots,
    _weekly_atr,
)

# ==============================
# Конфиги «по умолчанию»
# ==============================
TICKERS = os.getenv(
    "ARXORA_TRAIN_TICKERS", "AAPL,MSFT,TSLA,NVDA,SPY,QQQ,X:BTCUSD,X:ETHUSD"
).split(",")
MONTHS = int(os.getenv("ARXORA_TRAIN_MONTHS", "12"))
HORIZON = "Краткосрок (1–5 дней)"  # учим ST
MODEL_DIR = os.getenv("ARXORA_MODEL_DIR", "models")
OUT_PATH = os.path.join(MODEL_DIR, "arxora_lgbm_ST.joblib")

FILL_WINDOW = 3  # дней на касание entry
HOLD_DAYS = 5  # окно удержания для ST
MIN_ROWS_TO_TRAIN = 800  # нижняя граница на объём данных (можно уменьшить)
FEATS = [
    "pos",
    "slope_norm",
    "atr_d_over_price",
    "vol_ratio",
    "streak",
    "band",
    "long_upper",
    "long_lower",
]


def compute_levels_asof(df_asof: pd.DataFrame, horizon: str):
    """Уровни Entry/SL/TP1 в BUY-ветке (для разметки метки y)."""
    cfg = _horizon_cfg(horizon)
    price = float(df_asof["close"].iloc[-1])
    atr_d = float(_atr_like(df_asof, n=cfg["atr"]).iloc[-1])
    atr_w = _weekly_atr(df_asof) if cfg.get("use_weekly_atr") else atr_d

    hlc = _last_period_hlc(df_asof, cfg["pivot_period"])
    if not hlc:
        hlc = (
            float(df_asof["high"].tail(60).max()),
            float(df_asof["low"].tail(60).min()),
            float(df_asof["close"].iloc[-1]),
        )
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2 = piv["P"], piv["R1"], piv.get("R2")
    step_w = atr_w
    # Простая BUY-ветка
    if price < P:
        entry = max(price, piv["S1"] + 0.15 * step_w)
        sl = piv["S1"] - 0.60 * step_w
    else:
        entry = max(price, P + 0.10 * step_w)
        sl = P - 0.60 * step_w

    tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
    hz = _hz_tag(horizon)
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, "BUY", hz, price, step_w)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, "BUY")
    return entry, sl, tp1


def label_tp_before_sl(
    df: pd.DataFrame, start_ix: int, entry: float, sl: float, tp1: float, hold_days: int
) -> int:
    """Возвращает 1, если TP1 достигнут раньше SL в ближайшие hold_days после start_ix (включая его), иначе 0."""
    lo = df["low"].values
    hi = df["high"].values
    N = len(df)
    end = min(N, start_ix + hold_days)
    for k in range(start_ix, end):
        # первый удар решает исход
        if lo[k] <= sl:
            return 0
        if hi[k] >= tp1:
            return 1
    return 0


def build_dataset(tickers, horizon=HORIZON, months=MONTHS) -> pd.DataFrame:
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    look = cfg["look"]
    rows = []

    for t in tickers:
        try:
            days = int(months * 31) + look + 40
            df = cli.daily_ohlc(t, days=days).dropna()
            if len(df) < look + 30:
                print(f"[skip] {t}: мало данных ({len(df)})")
                continue

            for i in range(look + 5, len(df) - 6):
                df_asof = df.iloc[: i + 1]
                price = float(df_asof["close"].iloc[-1])

                # признаки — как в инференсе
                tail = df_asof.tail(look)
                rng_low, rng_high = float(tail["low"].min()), float(tail["high"].max())
                rng_w = max(1e-9, rng_high - rng_low)
                pos = (price - rng_low) / rng_w

                atr_d = float(_atr_like(df_asof, n=cfg["atr"]).iloc[-1])
                atr2 = float(_atr_like(df_asof, n=cfg["atr"] * 2).iloc[-1])
                vol_ratio = atr_d / max(1e-9, atr2)
                slope = _linreg_slope(df_asof["close"].tail(cfg["trend"]).values) / max(
                    1e-9, price
                )
                streak = _streak(df_asof["close"])

                hlc = _last_period_hlc(df_asof, cfg["pivot_period"])
                if not hlc:
                    hlc = (
                        float(df_asof["high"].tail(60).max()),
                        float(df_asof["low"].tail(60).min()),
                        float(df_asof["close"].iloc[-1]),
                    )
                piv = _fib_pivots(*hlc)
                band = _classify_band(
                    price,
                    piv,
                    0.25
                    * (_weekly_atr(df_asof) if cfg.get("use_weekly_atr") else atr_d),
                )

                # свечные «тени»
                lw_row = df_asof.iloc[-1]
                body = abs(lw_row["close"] - lw_row["open"])
                upper = max(0.0, lw_row["high"] - max(lw_row["open"], lw_row["close"]))
                lower = max(0.0, min(lw_row["open"], lw_row["close"]) - lw_row["low"])
                long_upper = (upper > body * 1.3) and (upper > lower * 1.1)
                long_lower = (lower > body * 1.3) and (lower > upper * 1.1)

                # уровни
                entry, sl, tp1 = compute_levels_asof(df_asof, horizon)

                # ждём касания entry в ближайшие FILL_WINDOW дней; если нет — пропускаем этот день
                touch_ix = None
                j_end = min(i + 1 + FILL_WINDOW, len(df))
                for j in range(i + 1, j_end):
                    lo = float(df["low"].iloc[j])
                    hi = float(df["high"].iloc[j])
                    if lo <= entry <= hi:
                        touch_ix = j
                        break
                if touch_ix is None:
                    continue

                y = label_tp_before_sl(df, touch_ix, entry, sl, tp1, HOLD_DAYS)

                rows.append(
                    dict(
                        ticker=t,
                        date=df_asof.index[-1].date(),
                        y=int(y),
                        pos=pos,
                        slope_norm=slope,
                        atr_d_over_price=atr_d / max(1e-9, price),
                        vol_ratio=vol_ratio,
                        streak=float(streak),
                        band=float(band),
                        long_upper=float(long_upper),
                        long_lower=float(long_lower),
                    )
                )
        except Exception as e:
            print(f"[error] {t}: {e}")

    return pd.DataFrame(rows)


def train_and_save(df: pd.DataFrame, out_path: str):
    if len(df) < MIN_ROWS_TO_TRAIN:
        print(
            f"[warn] Мало строк для обучения: {len(df)} (< {MIN_ROWS_TO_TRAIN}). Модель всё равно будет обучена, но качество может просесть."
        )

    X = df[FEATS].astype(float)
    y = df["y"].astype(int)
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # пытаемся LightGBM, иначе — LogisticRegression
    clf = None
    try:
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        clf.fit(Xtr, ytr)
        model_name = "LightGBM"
    except Exception:
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xtr, ytr)
        model_name = "LogReg(balanced)"

    # метрики
    try:
        proba = clf.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, proba)
    except Exception:
        pred = clf.predict(Xva)
        # у некоторых моделей может не быть predict_proba
        proba = pred.astype(float)
        auc = roc_auc_score(yva, proba) if len(np.unique(proba)) > 1 else float("nan")

    acc = accuracy_score(yva, (proba >= 0.5).astype(int))
    print(
        f"[{model_name}] val AUC={auc:.3f} · ACC(0.5)={acc:.3f} · rows={len(df)} · pos_rate={df['y'].mean():.3f}"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(clf, out_path)
    print("Saved model =>", out_path)


if __name__ == "__main__":
    print("Tickers:", TICKERS)
    print("Months:", MONTHS, "| Horizon:", HORIZON)
    if not os.getenv("POLYGON_API_KEY"):
        print("ERROR: не найден POLYGON_API_KEY в окружении (.env).")
        sys.exit(1)

    df = build_dataset(TICKERS, horizon=HORIZON, months=MONTHS)
    print(
        "Dataset shape:", df.shape, "| pos_rate:", df["y"].mean() if len(df) else "n/a"
    )

    if len(df) == 0:
        print(
            "Нет данных для обучения. Проверь TICKERS / API ключ / доступность истории."
        )
        sys.exit(1)

    train_and_save(df, OUT_PATH)
