#!/usr/bin/env python3
"""Simple backtest runner for Octopus"""

import sys
import os
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

# Проверяем наличие API ключа
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    raise ValueError(
        "POLYGON_API_KEY environment variable is not set. "
        "Please ensure it's configured in GitHub Actions secrets."
    )

print(f"Using API key: {POLYGON_API_KEY[:8]}***")

from core.strategy import analyze_asset_global

# Попробуйте разные форматы тикеров
TICKERS = [
    "X:BTCUSD",  # Polygon формат для криптовалют
    "X:ETHUSD",
    "AAPL",      # Обычные акции
    "NVDA"
]

def run_backtest():
    print("\n🚀 Starting Octopus backtest...\n")
    
    success_count = 0
    for ticker in TICKERS:
        print(f"📊 Testing {ticker.upper()}")
        try:
            result = analyze_asset_global(ticker, "Краткосрочный")
            action = result["recommendation"]["action"]
            conf = result["recommendation"]["confidence"]
            print(f"  ✅ Global: {action} (confidence: {conf:.2f})")
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Backtest complete! {success_count}/{len(TICKERS)} successful\n")

if __name__ == "__main__":
    run_backtest()
