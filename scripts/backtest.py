# scripts/backtest.py
import argparse, pandas as pd, numpy as np, datetime as dt
from pathlib import Path

from core.polygon_client import PolygonClient
from core.strategy import analyze_asset
from core.backtest_filters import dedupe_within_horizon

def generate_signals(ticker, horizon, days=800, warmup=180, conf_min=0.0):
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=days).sort_index()
    out = []
    for t in df.index[warmup:-1]:  # оставим хотя бы 1 день на оценку исхода
        df_slice = df.loc[:t]
        price_t  = float(df_slice["close"].iloc[-1])
        sig = analyze_asset(
            ticker=ticker, horizon=horizon,
            df_override=df_slice, price_override=price_t, ts=t
        )
        if sig["recommendation"]["action"] in ("BUY","SHORT") and \
           float(sig["recommendation"]["confidence"]) >= conf_min:
            sig["ticker"] = ticker
            out.append(sig)
    return out, df

def first_touch_outcome(future_df: pd.DataFrame, sig: dict) -> dict:
    """
    Возвращает исход трейда:
      {'touched': False/True, 'tp_hit': 0|1|2|3, 'rr': -1|0|1|2|3}
    Логика:
      - ждём касания entry (BUY: high>=entry; SHORT: low<=entry)
      - после входа идём по дням; если бар касается и SL, и TPi — считаем, что первым был SL (консервативно)
    """
    lv = sig["levels"]
    side = sig["recommendation"]["action"]
    entry, sl = float(lv["entry"]), float(lv["sl"])
    tp1, tp2, tp3 = float(lv["tp1"]), float(lv["tp2"]), float(lv["tp3"])

    # ждём вход
    i_enter = None
    for i, row in enumerate(future_df.itertuples(index=False)):
        h, l = float(row.high), float(row.low)
        if side == "BUY" and h >= entry:
            i_enter = i; break
        if side == "SHORT" and l <= entry:
            i_enter = i; break
    if i_enter is None:
        return {"touched": False, "tp_hit": 0, "rr": 0.0}

    path = future_df.iloc[i_enter:]
    for row in path.itertuples(index=False):
        h, l = float(row.high), float(row.low)
        if side == "BUY":
            # консервативный порядок проверки
            if l <= sl:           return {"touched": True, "tp_hit": 0, "rr": -1.0}
            if h >= tp3:          return {"touched": True, "tp_hit": 3, "rr":  3.0}
            if h >= tp2:          return {"touched": True, "tp_hit": 2, "rr":  2.0}
            if h >= tp1:          return {"touched": True, "tp_hit": 1, "rr":  1.0}
        else:  # SHORT
            if h >= sl:           return {"touched": True, "tp_hit": 0, "rr": -1.0}
            if l <= tp3:          return {"touched": True, "tp_hit": 3, "rr":  3.0}
            if l <= tp2:          return {"touched": True, "tp_hit": 2, "rr":  2.0}
            if l <= tp1:          return {"touched": True, "tp_hit": 1, "rr":  1.0}
    return {"touched": True, "tp_hit": 0, "rr": 0.0}  # никуда не дошли

def run_backtest(ticker, horizon, days, warmup, conf_min, cooldown_days, label="clean"):
    raw, df = generate_signals(ticker, horizon, days, warmup, conf_min)
    clean = dedupe_within_horizon(
        raw, horizon_key=horizon, by_day=True,
        tol_mult=0.5, tol_mode="risk", cooldown_days=cooldown_days
    )

    # симуляция исходов
    results = []
    for s in clean:
        ts = dt.datetime.fromisoformat(s["ts"])
        # будущие бары после ts
        fut = df.loc[df.index > ts]
        out = first_touch_outcome(fut, s)
        results.append({**s, **out})

    # агрегация
    rr = [r["rr"] for r in results]
    tp_hits = pd.Series([r["tp_hit"] for r in results]).value_counts().to_dict()
    touched = sum(1 for r in results if r["touched"])
    trades  = len(results)
    pnl_R   = float(np.sum(rr))
    winrate = 100.0 * sum(1 for x in rr if x > 0) / trades if trades else 0.0

    print(f"[{ticker} • {horizon}] raw={len(raw)}  clean={trades}  touched={touched}")
    print(f"TP hits: {tp_hits}   PnL (R): {pnl_R:.2f}   Win%: {winrate:.1f}%")

    # сохранить
    Path("backtests").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(f"backtests/{ticker}_{label}.csv", index=False)
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--horizon", default="Среднесрок (1–4 недели)")
    ap.add_argument("--days", type=int, default=800)
    ap.add_argument("--warmup", type=int, default=180)
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--cooldown", type=int, default=3)
    args = ap.parse_args()

    run_backtest(
        ticker=args.ticker,
        horizon=args.horizon,
        days=args.days,
        warmup=args.warmup,
        conf_min=args.conf,
        cooldown_days=args.cooldown
    )
