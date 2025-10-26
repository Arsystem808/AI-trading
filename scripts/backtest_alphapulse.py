#!/usr/bin/env python3
import json, os, subprocess, sys, glob, shutil, time
from datetime import datetime, timedelta
from pathlib import Path

def snapshot_jsons():
    return set(glob.glob("results/*.json"))

def newest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None

def main():
    Path("results").mkdir(exist_ok=True)
    end = datetime.utcnow().date(); start = end - timedelta(days=365)
    start_s = os.getenv("START_DATE", start.isoformat())
    end_s = os.getenv("END_DATE", end.isoformat())
    tickers = os.getenv("TICKERS", "AAPL,MSFT,NVDA,GOOGL,AMZN")

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    desired = f"results/alphapulse_backtest_{ts}.json"

    before = snapshot_jsons()
    cmd = [
        sys.executable, "scripts/run_backtest.py",
        "--strategy", "AlphaPulse",
        "--tickers", tickers,
        "--start", start_s,
        "--end", end_s,
        "--output", desired,
        "--format", "json"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    time.sleep(1)
    after = snapshot_jsons()
    produced = list(after - before)

    if Path(desired).exists():
        actual = desired
    elif produced:
        actual = produced[0]
        shutil.copyfile(actual, desired)
        actual = desired
    else:
        cand = newest("results/*backtest_*.json") or newest("results/*.json")
        if not cand:
            print("No backtest json produced in results/", file=sys.stderr); sys.exit(1)
        shutil.copyfile(cand, desired)
        actual = desired

    try:
        with open(actual) as f:
            data = json.load(f)
        print(json.dumps({
            "strategy": "AlphaPulse",
            "file": actual,
            "period": [start_s, end_s],
            "tickers": tickers.split(","),
            "summary": {
                "total_return_pct": data.get("total_return_pct"),
                "sharpe": data.get("sharpe"),
                "max_drawdown_pct": data.get("max_drawdown_pct"),
                "total_trades": data.get("total_trades"),
            }
        }, indent=2))
    except Exception as e:
        print(f"Warning: cannot read summary from {actual}: {e}")

    print(actual)

if __name__ == "__main__":
    main()
