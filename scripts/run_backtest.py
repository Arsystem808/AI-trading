#!/usr/bin/env python3
"""Simple backtest runner for Octopus"""

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.strategy import analyze_asset_global

TICKERS = ["btcusd", "ethusd"]

def run_backtest():
    print("Starting Octopus backtest...")
    
    for ticker in TICKERS:
        print(f"\nTesting {ticker.upper()}")
        try:
            result = analyze_asset_global(ticker, "Краткосрочный")
            action = result["recommendation"]["action"]
            conf = result["recommendation"]["confidence"]
            print(f"  Global: {action} (confidence: {conf:.2f})")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nBacktest complete!")

if __name__ == "__main__":
    run_backtest()
