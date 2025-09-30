# jobs/daily_benchmarks.py
from __future__ import annotations
import os, csv, argparse, time
from datetime import datetime, timezone
from typing import List, Dict, Any

# 1) ядро стратегий
from core.strategy import analyze_asset

# 2) мягкий импорт трекера эффективности
try:
    from core.performance_tracker import log_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass

DATA_DIR = os.environ.get("PERF_DATA_DIR", "data/perf")
AGENTS   = ["Global", "M7", "W7", "AlphaPulse", "Octopus"]
DEFAULT_TICKERS = os.environ.get("BENCH_TICKERS", "SPY,QQQ,DIA,IWM").split(",")
DEFAULT_HORIZON = os.environ.get("BENCH_HORIZON", "Краткосрочный")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _row_from_result(agent: str, ticker: str, horizon: str, res: Dict[str, Any]) -> Dict[str, Any]:
    rec   = res.get("recommendation", {}) or {}
    lev   = res.get("levels", {}) or {}
    probs = res.get("probs", {}) or {}
    now   = datetime.now(timezone.utc).isoformat()
    return {
        "ts": now,
        "agent": agent,
        "ticker": ticker,
        "horizon": horizon,
        "action": rec.get("action", ""),
        "confidence": float(rec.get("confidence", 0.0) or 0.0),
        "entry": float(lev.get("entry", 0.0) or 0.0),
        "sl": float(lev.get("sl", 0.0) or 0.0),
        "tp1": float(lev.get("tp1", 0.0) or 0.0),
        "tp2": float(lev.get("tp2", 0.0) or 0.0),
        "tp3": float(lev.get("tp3", 0.0) or 0.0),
        "p_tp1": float(probs.get("tp1", 0.0) or 0.0),
        "p_tp2": float(probs.get("tp2", 0.0) or 0.0),
        "p_tp3": float(probs.get("tp3", 0.0) or 0.0),
        # Зарезервировано под факты, если будет размечаться ex‑post:
        # "tp1_hit": "", "tp2_hit": "", "tp3_hit": "", "sl_hit": ""
    }

def _append_csv(path: str, row: Dict[str, Any]):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            ensure_dir(os.path.dirname(path))
            w.writeheader()
        w.writerow(row)

def run_once(tickers: List[str], horizon: str = DEFAULT_HORIZON, sleep_s: float = 0.8):
    ensure_dir(DATA_DIR)
    combined_path = os.path.join(DATA_DIR, "all.csv")
    for ticker in tickers:
        t = ticker.strip()
        if not t:
            continue
        for agent in AGENTS:
            try:
                res = analyze_asset(t, horizon, strategy=agent)
                row = _row_from_result(agent, t, horizon, res)
                _append_csv(os.path.join(DATA_DIR, f"{agent}.csv"), row)
                _append_csv(combined_path, row)
                # Дублирование в performance‑трекиер (если подключён)
                try:
                    log_agent_performance(
                        agent=agent,
                        ticker=t,
                        horizon=horizon,
                        action=row["action"],
                        confidence=row["confidence"],
                        levels={"entry": row["entry"], "sl": row["sl"], "tp1": row["tp1"], "tp2": row["tp2"], "tp3": row["tp3"]},
                        probs={"tp1": row["p_tp1"], "tp2": row["p_tp2"], "tp3": row["p_tp3"]},
                        meta={},
                        ts=row["ts"],
                    )
                except Exception:
                    pass
                print(f"[OK] {row['ts']} {agent} {t} {row['action']} {row['confidence']:.2f}")
            except Exception as e:
                print(f"[ERR] {agent} {t}: {e}")
            time.sleep(sleep_s)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="comma-separated list")
    ap.add_argument("--horizon", type=str, default=DEFAULT_HORIZON)
    ap.add_argument("--sleep", type=float, default=0.8)
    args = ap.parse_args()
    tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
    run_once(tickers, args.horizon, args.sleep)
