# 1. Создайте файл scripts/run_backtest.py
mkdir -p scripts
cat > scripts/run_backtest.py << 'EOF'
#!/usr/bin/env python3
"""Simple backtest runner for Octopus"""

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.strategy import analyze_asset_global, analyze_asset_m7, analyze_asset_w7

TICKERS = ["btcusd", "ethusd"]

def run_backtest():
    print("🚀 Starting Octopus backtest...")
    
    for ticker in TICKERS:
        print(f"\n📊 Testing {ticker.upper()}")
        try:
            # Простой вызов одной модели для проверки
            result = analyze_asset_global(ticker, "Краткосрочный")
            action = result["recommendation"]["action"]
            conf = result["recommendation"]["confidence"]
            print(f"  ✅ Global: {action} (confidence: {conf:.2f})")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    run_backtest()
EOF

# 2. Сделать исполняемым
chmod +x scripts/run_backtest.py

# 3. Закоммитить
git add scripts/run_backtest.py
git commit -m "Add backtest runner script"
git push
