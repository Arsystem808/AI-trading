#!/usr/bin/env python3
import json, os, subprocess, sys, glob, shutil
from datetime import datetime, timedelta
from pathlib import Path

def main():
    Path("results").mkdir(exist_ok=True)
    end = datetime.utcnow().date(); start = end - timedelta(days=365)
    start_s = os.getenv("START_DATE", start.isoformat())
    end_s = os.getenv("END_DATE", end.isoformat())
    tickers = os.getenv("TICKERS", "AAPL,MSFT,NVDA,GOOGL,AMZN")

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    desired = f"results/m7_backtest_{ts}.json"

    cmd = [
        sys.executable, "scripts/run_backtest.py",
        "--strategy", "M7",
        "--tickers", tickers,
        "--start", start_s,
        "--end", end_s,
        "--output", desired,      # если run_backtest.py игнорирует, ниже поймаем фактический путь
        "--format", "json"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 1) Если файл создан по ожидаемому пути — отлично
    if Path(desired).exists():
        actual = desired
    else:
        # 2) Иначе ищем последний json, который создал run_backtest.py (octopus_backtest_*.json и т.п.)
        candidates = sorted(glob.glob("results/*backtest_*.json"), key=os.path.getmtime, reverse=True)
        if not candidates:
            print("No backtest json produced in results/", file=sys.stderr)
            sys.exit(1)
        actual = candidates[0]
        # 3) Скопируем под нужное имя для консистентности артефактов
        shutil.copyfile(actual, desired)
        actual = desired

    # Опционально: вывести короткую сводку, если в JSON есть метрики
    try:
        with open(actual) as f:
            data = json.load(f)
        print(json.dumps({
            "strategy": "M7",
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
