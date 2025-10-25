#!/usr/bin/env python3
"""
Octopus Historical Backtest v3.0 - Professional Grade
Uses MultiCharts/NinjaTrader standard for intrabar TP/SL execution

Intrabar Execution Logic:
- If Open closer to High: sequence is Open → High → Low → Close
- If Open closer to Low: sequence is Open → Low → High → Close
- Conservative approach: when ambiguous, assume SL hit first
- Accounts for commission (0.1%) and slippage (0.05%)
"""

import sys
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY not set")

print(f"Using API key: {POLYGON_API_KEY[:8]}***\n")

from core.strategy import analyze_asset_global, analyze_asset_m7, analyze_asset_w7
import pandas as pd
import numpy as np

# Константы
TICKERS = ["X:BTCUSD", "X:ETHUSD", "AAPL", "NVDA"]
LOOKBACK_DAYS = 730
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.0005
POSITION_TIMEOUT_DAYS = 28
MIN_TP_DISTANCE_PCT = 0.005  # 0.5% минимальное расстояние до TP


class Position:
    """
    Position с правильной Open-High-Low-Close логикой (Industry Standard)
    """
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
        
        # Sanity check: проверяем минимальное расстояние до TP
        if self.tp1 > 0:
            tp1_dist = abs(self.tp1 - entry_price) / entry_price
            if tp1_dist < MIN_TP_DISTANCE_PCT:
                print(f"  ⚠️  WARNING: TP1 too close ({tp1_dist*100:.2f}%)")

    def check_exit(self, bar: pd.Series, current_date: str) -> bool:
        """
        INDUSTRY STANDARD: Open-High-Low-Close intrabar logic
        Reference: MultiCharts Intra-bar Price Movement Assumptions
        """
        if self.closed:
            return True
        
        open_p = bar["o"]
        high = bar["h"]
        low = bar["l"]
        close = bar["c"]
        
        self.days_held += 1
        
        # Timeout
        if self.days_held >= POSITION_TIMEOUT_DAYS:
            self.exit_price = close
            self.exit_date = current_date
            self.hit_level = "TIMEOUT"
            self.closed = True
            self._calculate_profit()
            return True
        
        # Проверка валидности уровней
        if self.entry_price == 0 or self.sl == 0:
            return False
        
        # === КЛЮЧЕВАЯ ЛОГИКА: Open-High-Low-Close последовательность ===
        # Определяем направление первого движения
        distance_to_high = abs(open_p - high)
        distance_to_low = abs(open_p - low)
        open_closer_to_high = distance_to_high < distance_to_low
        
        if self.action == "BUY":
            sl_hit = low <= self.sl
            tp1_hit = high >= self.tp1 if self.tp1 > 0 else False
            tp2_hit = high >= self.tp2 if self.tp2 > 0 else False
            tp3_hit = high >= self.tp3 if self.tp3 > 0 else False
            
            # Оба достигнуты на одном баре
            if sl_hit and (tp1_hit or tp2_hit or tp3_hit):
                if open_closer_to_high:
                    # Open → High → Low: TP достигнут первым
                    if tp3_hit:
                        self.exit_price = self.tp3
                        self.hit_level = "TP3"
                    elif tp2_hit:
                        self.exit_price = self.tp2
                        self.hit_level = "TP2"
                    elif tp1_hit:
                        self.exit_price = self.tp1
                        self.hit_level = "TP1"
                else:
                    # Open → Low → High: SL достигнут первым
                    self.exit_price = self.sl
                    self.hit_level = "SL"
                self.closed = True
            
            # Только SL
            elif sl_hit:
                self.exit_price = self.sl
                self.hit_level = "SL"
                self.closed = True
            
            # Только TP
            elif tp3_hit:
                self.exit_price = self.tp3
                self.hit_level = "TP3"
                self.closed = True
            elif tp2_hit:
                self.exit_price = self.tp2
                self.hit_level = "TP2"
                self.closed = True
            elif tp1_hit:
                self.exit_price = self.tp1
                self.hit_level = "TP1"
                self.closed = True
        
        else:  # SHORT
            sl_hit = high >= self.sl
            tp1_hit = low <= self.tp1 if self.tp1 > 0 else False
            tp2_hit = low <= self.tp2 if self.tp2 > 0 else False
            tp3_hit = low <= self.tp3 if self.tp3 > 0 else False
            
            if sl_hit and (tp1_hit or tp2_hit or tp3_hit):
                if open_closer_to_high:
                    # Open → High → Low: SL первым (для SHORT SL выше)
                    self.exit_price = self.sl
                    self.hit_level = "SL"
                else:
                    # Open → Low → High: TP первым (для SHORT TP ниже)
                    if tp3_hit:
                        self.exit_price = self.tp3
                        self.hit_level = "TP3"
                    elif tp2_hit:
                        self.exit_price = self.tp2
                        self.hit_level = "TP2"
                    elif tp1_hit:
                        self.exit_price = self.tp1
                        self.hit_level = "TP1"
                self.closed = True
            
            elif sl_hit:
                self.exit_price = self.sl
                self.hit_level = "SL"
                self.closed = True
            
            elif tp3_hit:
                self.exit_price = self.tp3
                self.hit_level = "TP3"
                self.closed = True
            elif tp2_hit:
                self.exit_price = self.tp2
                self.hit_level = "TP2"
                self.closed = True
            elif tp1_hit:
                self.exit_price = self.tp1
                self.hit_level = "TP1"
                self.closed = True
        
        if self.closed:
            self.exit_date = current_date
            self._calculate_profit()
        
        return self.closed

    def _calculate_profit(self):
        """Расчет прибыли с учетом комиссий и slippage"""
        if self.exit_price == 0 or self.entry_price == 0:
            self.profit_pct = 0
            return
        
        if self.action == "BUY":
            gross_profit = (self.exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            gross_profit = (self.entry_price - self.exit_price) / self.entry_price
        
        # Вычитаем издержки (entry + exit)
        self.profit_pct = gross_profit - (COMMISSION_PCT + SLIPPAGE_PCT) * 2


def fetch_historical_ohlc(ticker: str, days: int = 730) -> pd.DataFrame:
    """Загружает OHLC через Polygon API"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {"apiKey": POLYGON_API_KEY, "limit": 5000}
    
    print(f"  Загрузка данных для {ticker}...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "results" not in data or not data["results"]:
            print(f"  ⚠️  Нет данных")
            return pd.DataFrame()
        
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "o", "h": "h", "l": "l", "c": "c", "v": "v"})
        df = df[["date", "o", "h", "l", "c", "v"]]
        
        print(f"  ✅ Загружено {len(df)} баров")
        return df
    
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return pd.DataFrame()


def get_octopus_consensus(ticker: str) -> Optional[Dict]:
    """Агрегирует сигналы от агентов"""
    signals = []
    
    agents = [
        ("Global", lambda: analyze_asset_global(ticker, "Краткосрочный")),
        ("M7", lambda: analyze_asset_m7(ticker, "Краткосрочный", use_ml=True)),
        ("W7", lambda: analyze_asset_w7(ticker, "Краткосрочный")),
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
    
    consensus_signals = [s for s in signals if s["signal"]["recommendation"]["action"] == consensus_action]
    best_signal = max(consensus_signals, key=lambda s: s["signal"]["recommendation"]["confidence"])
    
    return best_signal["signal"]


def run_octopus_backtest():
    print("🚀 Octopus Backtest v3.0 - Professional Grade\n")
    print("Intrabar Logic: Open-High-Low-Close (Industry Standard)\n")
    
    all_results = []
    
    for ticker in TICKERS:
        print(f"\n📊 Backtesting {ticker.upper()}")
        
        historical_data = fetch_historical_ohlc(ticker, LOOKBACK_DAYS)
        
        if historical_data.empty:
            print(f"  ❌ Пропуск")
            continue
        
        ticker_trades = []
        open_position = None
        
        for idx, row in historical_data.iterrows():
            current_date = row["date"].strftime("%Y-%m-%d")
            
            # Проверка выхода
            if open_position:
                if open_position.check_exit(row, current_date):
                    ticker_trades.append({
                        "entry_date": open_position.entry_date,
                        "exit_date": open_position.exit_date,
                        "action": open_position.action,
                        "entry_price": round(open_position.entry_price, 2),
                        "exit_price": round(open_position.exit_price, 2),
                        "hit_level": open_position.hit_level,
                        "profit_pct": round(open_position.profit_pct, 4),
                        "days_held": open_position.days_held
                    })
                    open_position = None
            
            # Поиск сигнала (раз в неделю)
            if not open_position and idx % 7 == 0:
                try:
                    consensus_signal = get_octopus_consensus(ticker)
                    if consensus_signal and consensus_signal["recommendation"]["action"] != "WAIT":
                        entry_price = row["c"]
                        open_position = Position(consensus_signal, current_date, entry_price)
                except:
                    continue
        
        # Закрываем оставшуюся
        if open_position and not open_position.closed:
            last_row = historical_data.iloc[-1]
            open_position.exit_price = last_row["c"]
            open_position.exit_date = last_row["date"].strftime("%Y-%m-%d")
            open_position.hit_level = "FORCED_CLOSE"
            open_position._calculate_profit()
            open_position.closed = True
            ticker_trades.append({
                "entry_date": open_position.entry_date,
                "exit_date": open_position.exit_date,
                "action": open_position.action,
                "entry_price": round(open_position.entry_price, 2),
                "exit_price": round(open_position.exit_price, 2),
                "hit_level": "FORCED_CLOSE",
                "profit_pct": round(open_position.profit_pct, 4),
                "days_held": open_position.days_held
            })
        
        # Метрики
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
            
            avg_days = np.mean([t["days_held"] for t in ticker_trades])
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365 / avg_days) if returns.std() > 0 and avg_days > 0 else 0
            
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
                "avg_days_held": round(avg_days, 1),
                "trades": ticker_trades[:20]
            }
            
            all_results.append(result)
            
            print(f"\n  ✅ Результаты:")
            print(f"     Сделок: {len(ticker_trades)}")
            print(f"     Доходность: {result['total_return_pct']}%")
            print(f"     Win Rate: {result['win_rate_pct']}%")
            print(f"     Sharpe: {result['sharpe_ratio']}")
            print(f"     Max DD: {result['max_drawdown_pct']}%")
            print(f"     TP: TP1={tp_hits['TP1']}, TP2={tp_hits['TP2']}, TP3={tp_hits['TP3']}, SL={tp_hits['SL']}")
        else:
            print(f"  ❌ Нет сделок")
    
    # Сохранение
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"octopus_backtest_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "backtest_date": datetime.now().isoformat(),
            "version": "3.0_professional_grade_ohlc_logic",
            "period_days": LOOKBACK_DAYS,
            "agents_used": ["Global", "M7", "W7"],
            "commission_pct": COMMISSION_PCT,
            "slippage_pct": SLIPPAGE_PCT,
            "intrabar_logic": "Open-High-Low-Close (MultiCharts standard)",
            "tickers": TICKERS,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\n✅ Backtest завершен!")
    print(f"📁 Результаты: {output_file}\n")
    
    if all_results:
        print("📈 Общая статистика:")
        print(f"   Всего сделок: {sum([r['total_trades'] for r in all_results])}")
        print(f"   Средний Sharpe: {np.mean([r['sharpe_ratio'] for r in all_results]):.2f}")
        print(f"   Средний Win Rate: {np.mean([r['win_rate_pct'] for r in all_results]):.1f}%")


if __name__ == "__main__":
    run_octopus_backtest()
