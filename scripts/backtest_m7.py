#!/usr/bin/env python3
import json, os, subprocess, sys
from datetime import datetime, timedelta
from pathlib import Path

def env_date(name, default):
    v = os.getenv(name)
    return v if v else default

def main():
    Path("results").mkdir(exist_ok=True)
    end = datetime.utcnow().date()
    start = end - timedelta(days=365)
    start_s = env_date("START_DATE", start.isoformat())
    end_s = env_date("END_DATE", end.isoformat())
    tickers = os.getenv("TICKERS", "AAPL,MSFT,NVDA,GOOGL,AMZN")

    out_path = f"results/m7_backtest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    cmd = [
        sys.executable, "scripts/run_backtest.py",
        "--strategy", "M7",
        "--tickers", tickers,
        "--start", start_s,
        "--end", end_s,
        "--output", out_path,
        "--format", "json"
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Backtest failed:", e, file=sys.stderr)
        sys.exit(1)

    with open(out_path) as f:
        data = json.load(f)
    print(out_path)
    print(json.dumps({
        "strategy": "M7",
        "file": out_path,
        "period": [start_s, end_s],
        "tickers": tickers.split(","),
        "summary": {
            "total_return_pct": data.get("total_return_pct"),
            "sharpe": data.get("sharpe"),
            "max_drawdown_pct": data.get("max_drawdown_pct"),
            "total_trades": data.get("total_trades"),
        }
    }, indent=2))

if __name__ == "__main__":
    main()
