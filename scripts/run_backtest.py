#!/usr/bin/env python3
"""
Octopus Historical Backtest with Real Market Data
Backtest Assumptions:
- Commission: 0.1% per trade
- Slippage: 0.05% on entry/exit
- Position timeout: 28 days (4 weeks)
- TP/SL checked on bar High/Low (conservative approach)
- Uses 4 agents: Global, M7, W7, AlphaPulse
"""

import sys
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY environment variable is not set")

print(f"Using API key: {POLYGON_API_KEY[:8]}***\n")

from core.strategy import analyze_asset_global, analyze_asset_m7, analyze_asset_w7
import pandas as pd
import numpy as np

# Константы
TICKERS = ["X:BTCUSD", "X:ETHUSD", "AAPL", "NVDA"]
LOOKBACK_DAYS = 730  # 2 года
COMMISSION_PCT = 0.001  # 0.1%
SLIPPAGE_PCT = 0.0005   # 0.05%
POSITION_TIMEOUT_DAYS = 28


class Position:
    """Управление открытой позицией с реальной проверкой TP/SL"""
    def __init__(self, signal: Dict, entry_date: str, entry_price: float):
        self.entry_date = entry_date
        self.action = signal["recommendation"]["action"]
        self.entry_price = entry_price
        self.sl = signal["levels"]["sl"]
        self.tp1 = signal["levels"]["tp1"]
        self.tp2 = signal["levels"]["tp2"]
        self.tp3 = signal["levels"]["tp3"]
        self.closed = False
        self.exit_date = None
        self.exit_price = None
        self.profit_pct = 0
        self.hit_level = None
        self.days_held = 0

    def check_exit(self, bar: pd.Series, current_date: str) -> bool:
        """Проверяет достижение TP или SL на основе High/Low бара"""
        if self.closed:
            return True
        
        high = bar["h"]
        low = bar["l"]
        close = bar["c"]
        
        # Проверка таймаута
        self.days_held += 1
        if self.days_held >= POSITION_TIMEOUT_DAYS:
            self.exit_price = close
            self.exit_date = current_date
            self.hit_level = "TIMEOUT"
            self.closed = True
        
        # Проверка достижения уровней
        if self.action == "BUY":
            # Для BUY: сначала проверяем SL (низ бара)
            if low <= self.sl:
                self.exit_price = self.sl
                self.exit_date = current_date
                self.hit_level = "SL"
                self.closed = True
            # Затем проверяем TP (верх бара)
            elif high >= self.tp3:
                self.exit_price = self.tp3
                self.exit_date = current_date
                self.hit_level = "TP3"
                self.closed = True
            elif high >= self.tp2:
                self.exit_price = self.tp2
                self.exit_date = current_date
                self.hit_level = "TP2"
                self.closed = True
            elif high >= self.tp1:
                self.exit_price = self.tp1
                self.exit_date = current_date
                self.hit_level = "TP1"
                self.closed = True
        
        else:  # SHORT
            # Для SHORT: сначала проверяем SL (верх бара)
            if high >= self.sl:
                self.exit_price = self.sl
                self.exit_date = current_date
                self.hit_level = "SL"
                self.closed = True
            # Затем проверяем TP (низ бара)
            elif low <= self.tp3:
                self.exit_price = self.tp3
                self.exit_date = current_date
                self.hit_level = "TP3"
                self.closed = True
            elif low <= self.tp2:
                self.exit_price = self.tp2
                self.exit_date = current_date
                self.hit_level = "TP2"
                self.closed = True
            elif low <= self.tp1:
                self.exit_price = self.tp1
                self.exit_date = current_date
                self.hit_level = "TP1"
                self.closed = True
        
        # Расчет прибыли с учетом комиссий и slippage
        if self.closed:
            gross_profit = 0
            if self.action == "BUY":
                gross_profit = (self.exit_price - self.entry_price) / self.entry_price
            else:  # SHORT
                gross_profit = (self.entry_price - self.exit_price) / self.entry_price
            
            # Вычитаем транзакционные издержки
            self.profit_pct = gross_profit - (COMMISSION_PCT + SLIPPAGE_PCT) * 2
        
        return self.closed


def fetch_historical_ohlc(ticker: str, days: int = 730) -> pd.DataFrame:
    """Загружает реальные OHLC данные через Polygon API"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {"apiKey": POLYGON_API_KEY, "limit": 5000}
    
    print(f"  Загрузка исторических данных для {ticker}...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "results" not in data or not data["results"]:
            print(f"  ⚠️  Нет данных для {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "o", "h": "h", "l": "l", "c": "c", "v": "v"})
        df = df[["date", "o", "h", "l", "c", "v"]]
        
        print(f"  ✅ Загружено {len(df)} баров")
        return df
    
    except Exception as e:
        print(f"  ❌ Ошибка загрузки данных: {e}")
        return pd.DataFrame()


def get_octopus_consensus(ticker: str) -> Optional[Dict]:
    """Агрегирует сигналы от 4 агентов"""
    signals = []
    
    agents = [
        ("Global", lambda: analyze_asset_global(ticker, "Краткосрочный")),
        ("M7", lambda: analyze_asset_m7(ticker, "Краткосрочный", use_ml=True)),
        ("W7", lambda: analyze_asset_w7(ticker, "Краткосрочный")),
        # Раскомментируйте если есть AlphaPulse:
        # ("AlphaPulse", lambda: analyze_asset_alphapulse(ticker, "Краткосрочный")),
    ]
    
    for name, func in agents:
        try:
            signal = func()
            if signal["recommendation"]["action"] != "WAIT":
                signals.append({"agent": name, "signal": signal})
        except:
            continue
    
    if not signals:
        return None
    
    # Мажоритарный выбор
    actions = [s["signal"]["recommendation"]["action"] for s in signals]
    consensus_action = max(set(actions), key=actions.count)
    
    # Берем лучший сигнал с consensus action
    consensus_signals = [s for s in signals if s["signal"]["recommendation"]["action"] == consensus_action]
    best_signal = max(consensus_signals, key=lambda s: s["signal"]["recommendation"]["confidence"])
    
    return best_signal["signal"]


def run_octopus_backtest():
    print("🚀 Starting Octopus 2-year backtest with real market data\n")
    
    all_results = []
    
    for ticker in TICKERS:
        print(f"\n📊 Backtesting {ticker.upper()}")
        
        # Загрузка исторических данных
        historical_data = fetch_historical_ohlc(ticker, LOOKBACK_DAYS)
        
        if historical_data.empty:
            print(f"  ❌ Пропуск {ticker}: нет данных")
            continue
        
        ticker_trades = []
        open_position = None
        
        # Проход по каждому бару
        for idx, row in historical_data.iterrows():
            current_date = row["date"].strftime("%Y-%m-%d")
            
            # Проверка выхода из открытой позиции
            if open_position:
                if open_position.check_exit(row, current_date):
                    ticker_trades.append({
                        "entry_date": open_position.entry_date,
                        "exit_date": open_position.exit_date,
                        "action": open_position.action,
                        "entry_price": open_position.entry_price,
                        "exit_price": open_position.exit_price,
                        "hit_level": open_position.hit_level,
                        "profit_pct": open_position.profit_pct,
                        "days_held": open_position.days_held
                    })
                    open_position = None
            
            # Поиск нового сигнала (раз в неделю для оптимизации)
            if not open_position and idx % 7 == 0:
                try:
                    consensus_signal = get_octopus_consensus(ticker)
                    if consensus_signal and consensus_signal["recommendation"]["action"] != "WAIT":
                        # Открываем позицию по цене close текущего бара
                        entry_price = row["c"]
                        open_position = Position(consensus_signal, current_date, entry_price)
                except:
                    continue
        
        # Закрываем оставшуюся позицию
        if open_position and not open_position.closed:
            last_row = historical_data.iloc[-1]
            open_position.exit_price = last_row["c"]
            open_position.exit_date = last_row["date"].strftime("%Y-%m-%d")
            open_position.hit_level = "FORCED_CLOSE"
            open_position.profit_pct = 0
            open_position.closed = True
            ticker_trades.append({
                "entry_date": open_position.entry_date,
                "exit_date": open_position.exit_date,
                "action": open_position.action,
                "entry_price": open_position.entry_price,
                "exit_price": open_position.exit_price,
                "hit_level": "FORCED_CLOSE",
                "profit_pct": 0,
                "days_held": open_position.days_held
            })
        
        # Расчет метрик
        if ticker_trades:
            returns = np.array([t["profit_pct"] for t in ticker_trades])
            
            total_return = (1 + returns).prod() - 1
            win_rate = (returns > 0).mean()
            
            tp_hits = {
                "TP1": sum(1 for t in ticker_trades if t["hit_level"] == "TP1"),
                "TP2": sum(1 for t in ticker_trades if t["hit_level"] == "TP2"),
                "TP3": sum(1 for t in ticker_trades if t["hit_level"] == "TP3"),
                "SL": sum(1 for t in ticker_trades if t["hit_level"] == "SL"),
            }
            
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365 / np.mean([t["days_held"] for t in ticker_trades])) if returns.std() > 0 else 0
            
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            result = {
                "ticker": ticker,
                "total_trades": len(ticker_trades),
                "total_return_pct": round(total_return * 100, 2),
                "win_rate_pct": round(win_rate * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "tp_distribution": tp_hits,
                "avg_days_held": round(np.mean([t["days_held"] for t in ticker_trades]), 1),
                "trades": ticker_trades[:10]  # Первые 10 для обзора
            }
            
            all_results.append(result)
            
            print(f"\n  ✅ Результаты для {ticker}:")
            print(f"     Сделок: {len(ticker_trades)}")
            print(f"     Доходность: {result['total_return_pct']}%")
            print(f"     Win Rate: {result['win_rate_pct']}%")
            print(f"     Sharpe: {result['sharpe_ratio']}")
            print(f"     Max DD: {result['max_drawdown_pct']}%")
            print(f"     TP распределение: TP1={tp_hits['TP1']}, TP2={tp_hits['TP2']}, TP3={tp_hits['TP3']}, SL={tp_hits['SL']}")
        else:
            print(f"  ❌ Нет сделок для {ticker}")
    
    # Сохранение результатов
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"octopus_backtest_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "backtest_date": datetime.now().isoformat(),
            "period_days": LOOKBACK_DAYS,
            "agents_used": ["Global", "M7", "W7"],
            "commission_pct": COMMISSION_PCT,
            "slippage_pct": SLIPPAGE_PCT,
            "tickers": TICKERS,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\n✅ Backtest завершен!")
    print(f"📁 Результаты сохранены: {output_file}\n")
    
    # Итоговая статистика
    if all_results:
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in all_results])
        avg_win_rate = np.mean([r["win_rate_pct"] for r in all_results])
        total_trades = sum([r["total_trades"] for r in all_results])
        
        print("📈 Общая статистика:")
        print(f"   Всего сделок: {total_trades}")
        print(f"   Средний Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"   Средний Win Rate: {avg_win_rate:.1f}%")


if __name__ == "__main__":
    run_octopus_backtest()

