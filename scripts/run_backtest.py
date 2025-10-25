#!/usr/bin/env python3
"""
Octopus Historical Backtest v7.0 - PRODUCTION PERFECTION
Real Octopus orchestrator + All best practices implemented

New in v7.0:
- Backtest mode flag (disables performance logging)
- ML models availability check
- WAIT rate monitoring and warnings
- Enhanced error handling and logging
"""

import sys
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

# === BACKTEST MODE FLAG ===
os.environ["BACKTEST_MODE"] = "1"
print("🔧 Backtest mode enabled (performance logging disabled)\n")

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY not set")

print(f"Using API key: {POLYGON_API_KEY[:8]}***\n")

from core.strategy import analyze_asset_octopus
import pandas as pd
import numpy as np

# Константы
TICKERS = ["X:BTCUSD", "X:ETHUSD", "AAPL", "NVDA"]
LOOKBACK_DAYS = 730
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.0005
POSITION_TIMEOUT_DAYS = 28
INITIAL_CAPITAL = 100000
RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_PCT = 0.10


def check_ml_models():
    """Проверяем наличие ML моделей"""
    print("🔍 Checking ML models availability...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("  ⚠️  models/ directory not found")
        print("  ℹ️  M7 agent will run without ML\n")
        return False
    
    # Проверяем основные модели
    expected_models = [
        models_dir / "arxora_m7pro" / f"{ticker.replace(':', '_')}_model.joblib"
        for ticker in TICKERS
    ]
    
    missing = [m for m in expected_models if not m.exists()]
    
    if missing:
        print(f"  ⚠️  Missing {len(missing)}/{len(expected_models)} ML models")
        print(f"  ℹ️  M7 agent will run without ML where models are missing\n")
        return False
    else:
        print(f"  ✅ All ML models found ({len(expected_models)} models)\n")
        return True


class Account:
    """Аккаунт с proper position sizing"""
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = [initial_capital]
        self.trades_log = []
    
    def calculate_position_size(self, entry_price: float, sl_price: float) -> float:
        """Position sizing: риск 1% капитала"""
        if entry_price == 0 or sl_price == 0:
            return 0
        
        risk_amount = self.equity * RISK_PER_TRADE_PCT
        price_risk_pct = abs((entry_price - sl_price) / entry_price)
        
        if price_risk_pct == 0:
            return 0
        
        shares = risk_amount / (entry_price * price_risk_pct)
        position_value = shares * entry_price
        
        max_position_value = self.equity * MAX_POSITION_PCT
        if position_value > max_position_value:
            position_value = max_position_value
            shares = position_value / entry_price
        
        return shares
    
    def execute_trade(self, trade: Dict):
        """Обновляем equity"""
        profit_loss = trade["profit_pct"] * trade["position_value"]
        self.equity += profit_loss
        self.equity_curve.append(self.equity)
        self.trades_log.append(trade)


class Position:
    """Position с Open-High-Low-Close логикой"""
    def __init__(self, signal: Dict, entry_date: str, entry_price: float, shares: float):
        self.entry_date = entry_date
        self.action = signal["recommendation"]["action"]
        self.entry_price = entry_price
        self.shares = shares
        self.position_value = entry_price * shares
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
        
        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {self.entry_price}")

    def check_exit(self, bar: pd.Series, current_date: str) -> bool:
        """Open-High-Low-Close intrabar logic"""
        if self.closed:
            return True
        
        open_p = bar["o"]
        high = bar["h"]
        low = bar["l"]
        close = bar["c"]
        
        if open_p <= 0 or high <= 0 or low <= 0 or close <= 0:
            return False
        
        self.days_held += 1
        
        if self.days_held >= POSITION_TIMEOUT_DAYS:
            self.exit_price = close
            self.exit_date = current_date
            self.hit_level = "TIMEOUT"
            self.closed = True
            self._calculate_profit()
            return True
        
        if self.entry_price == 0 or self.sl == 0:
            return False
        
        distance_to_high = abs(open_p - high)
        distance_to_low = abs(open_p - low)
        open_closer_to_high = distance_to_high < distance_to_low
        
        if self.action == "BUY":
            sl_hit = low <= self.sl
            tp1_hit = high >= self.tp1 if self.tp1 > 0 else False
            tp2_hit = high >= self.tp2 if self.tp2 > 0 else False
            tp3_hit = high >= self.tp3 if self.tp3 > 0 else False
            
            if sl_hit and (tp1_hit or tp2_hit or tp3_hit):
                if open_closer_to_high:
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
                    self.exit_price = self.sl
                    self.hit_level = "SL"
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
        
        else:  # SHORT
            sl_hit = high >= self.sl
            tp1_hit = low <= self.tp1 if self.tp1 > 0 else False
            tp2_hit = low <= self.tp2 if self.tp2 > 0 else False
            tp3_hit = low <= self.tp3 if self.tp3 > 0 else False
            
            if sl_hit and (tp1_hit or tp2_hit or tp3_hit):
                if open_closer_to_high:
                    self.exit_price = self.sl
                    self.hit_level = "SL"
                else:
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
        """Расчет без caps"""
        if self.exit_price == 0 or self.entry_price == 0:
            self.profit_pct = 0
            return
        
        if self.action == "BUY":
            gross_profit = (self.exit_price - self.entry_price) / self.entry_price
        else:
            gross_profit = (self.entry_price - self.exit_price) / self.entry_price
        
        self.profit_pct = gross_profit - (COMMISSION_PCT + SLIPPAGE_PCT) * 2


def fetch_historical_ohlc(ticker: str, days: int = 730) -> pd.DataFrame:
    """Загружает OHLC"""
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


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Max drawdown без caps"""
    if len(equity_curve) <= 1:
        return 0.0
    
    equity = np.array(equity_curve)
    
    if equity.min() <= 0:
        return -1.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    
    return drawdown.min()


def run_octopus_backtest():
    print("🚀 Octopus Backtest v7.0 - PRODUCTION PERFECTION\n")
    
    # Проверка ML моделей
    ml_available = check_ml_models()
    
    print(f"""
📝 Octopus Configuration:
   Orchestrator: analyze_asset_octopus()
   Agents: Global (0.13), M7 (0.20), W7 (0.26), AlphaPulse (0.28)
   
   Voting Rules:
   - ≥3 agents agree → Strong signal
   - Ratio < 0.20 → WAIT (conservative by design)
   
   Risk Management:
   - Initial Capital: ${INITIAL_CAPITAL:,}
   - Risk per Trade: {RISK_PER_TRADE_PCT*100}%
   - Max Position: {MAX_POSITION_PCT*100}%
   
   Expected WAIT Rates:
   - Crypto: 50-70% (high volatility)
   - Stocks: 30-50% (more stable)
""")
    
    all_results = []
    signal_stats = {"total_checks": 0, "wait": 0, "buy": 0, "short": 0}
    
    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"📊 Backtesting {ticker.upper()}")
        print(f"{'='*60}")
        
        historical_data = fetch_historical_ohlc(ticker, LOOKBACK_DAYS)
        
        if historical_data.empty:
            print(f"  ❌ Пропуск")
            continue
        
        account = Account(INITIAL_CAPITAL)
        open_position = None
        ticker_signal_stats = {"checks": 0, "wait": 0, "buy": 0, "short": 0}
        
        for idx, row in historical_data.iterrows():
            current_date = row["date"].strftime("%Y-%m-%d")
            
            if open_position:
                if open_position.check_exit(row, current_date):
                    trade = {
                        "entry_date": open_position.entry_date,
                        "exit_date": open_position.exit_date,
                        "action": open_position.action,
                        "entry_price": round(open_position.entry_price, 2),
                        "exit_price": round(open_position.exit_price, 2),
                        "shares": round(open_position.shares, 2),
                        "position_value": round(open_position.position_value, 2),
                        "hit_level": open_position.hit_level,
                        "profit_pct": round(open_position.profit_pct, 4),
                        "days_held": open_position.days_held
                    }
                    account.execute_trade(trade)
                    open_position = None
            
            # Поиск сигнала через Octopus
            if not open_position and idx % 7 == 0 and idx > 0:
                ticker_signal_stats["checks"] += 1
                signal_stats["total_checks"] += 1
                
                try:
                    octopus_signal = analyze_asset_octopus(ticker, "Краткосрочный")
                    action = octopus_signal["recommendation"]["action"]
                    
                    # Статистика сигналов
                    if action == "WAIT":
                        ticker_signal_stats["wait"] += 1
                        signal_stats["wait"] += 1
                    elif action == "BUY":
                        ticker_signal_stats["buy"] += 1
                        signal_stats["buy"] += 1
                    elif action == "SHORT":
                        ticker_signal_stats["short"] += 1
                        signal_stats["short"] += 1
                    
                    if action != "WAIT":
                        if idx + 1 < len(historical_data):
                            next_row = historical_data.iloc[idx + 1]
                            entry_price = next_row["o"]
                            sl_price = octopus_signal["levels"]["sl"]
                            
                            if entry_price > 0 and sl_price > 0:
                                shares = account.calculate_position_size(entry_price, sl_price)
                                if shares > 0:
                                    open_position = Position(octopus_signal, current_date, entry_price, shares)
                except Exception as e:
                    print(f"  ⚠️  Octopus error at {current_date}: {str(e)[:50]}")
                    continue
        
        # Закрываем оставшуюся
        if open_position and not open_position.closed:
            last_row = historical_data.iloc[-1]
            open_position.exit_price = last_row["c"]
            open_position.exit_date = last_row["date"].strftime("%Y-%m-%d")
            open_position.hit_level = "FORCED_CLOSE"
            open_position._calculate_profit()
            open_position.closed = True
            trade = {
                "entry_date": open_position.entry_date,
                "exit_date": open_position.exit_date,
                "action": open_position.action,
                "entry_price": round(open_position.entry_price, 2),
                "exit_price": round(open_position.exit_price, 2),
                "shares": round(open_position.shares, 2),
                "position_value": round(open_position.position_value, 2),
                "hit_level": "FORCED_CLOSE",
                "profit_pct": round(open_position.profit_pct, 4),
                "days_held": open_position.days_held
            }
            account.execute_trade(trade)
        
        # Метрики
        if account.trades_log:
            trades = account.trades_log
            returns_pct = np.array([t["profit_pct"] for t in trades])
            
            total_return = (account.equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
            win_rate = (returns_pct > 0).mean()
            
            tp_hits = {
                "TP1": sum(1 for t in trades if t["hit_level"] == "TP1"),
                "TP2": sum(1 for t in trades if t["hit_level"] == "TP2"),
                "TP3": sum(1 for t in trades if t["hit_level"] == "TP3"),
                "SL": sum(1 for t in trades if t["hit_level"] == "SL"),
            }
            
            avg_days = np.mean([t["days_held"] for t in trades])
            sharpe = (returns_pct.mean() / returns_pct.std()) * np.sqrt(365 / avg_days) if returns_pct.std() > 0 and avg_days > 0 else 0
            
            max_drawdown = calculate_max_drawdown(account.equity_curve)
            
            # WAIT rate для тикера
            wait_rate = ticker_signal_stats["wait"] / max(1, ticker_signal_stats["checks"])
            
            result = {
                "ticker": ticker,
                "total_trades": len(trades),
                "final_equity": round(account.equity, 2),
                "total_return_pct": round(total_return * 100, 2),
                "win_rate_pct": round(win_rate * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "tp_distribution": tp_hits,
                "avg_days_held": round(avg_days, 1),
                "signal_stats": ticker_signal_stats,
                "wait_rate_pct": round(wait_rate * 100, 1),
                "trades": trades[:20]
            }
            
            all_results.append(result)
            
            print(f"\n  ✅ Результаты:")
            print(f"     Сделок: {len(trades)}")
            print(f"     Final Equity: ${result['final_equity']:,.2f}")
            print(f"     Доходность: {result['total_return_pct']}%")
            print(f"     Win Rate: {result['win_rate_pct']}%")
            print(f"     Sharpe: {result['sharpe_ratio']}")
            print(f"     Max DD: {result['max_drawdown_pct']}%")
            print(f"     TP: TP1={tp_hits['TP1']}, TP2={tp_hits['TP2']}, TP3={tp_hits['TP3']}, SL={tp_hits['SL']}")
            print(f"     WAIT Rate: {result['wait_rate_pct']}% ({ticker_signal_stats['wait']}/{ticker_signal_stats['checks']} checks)")
            
            # Warnings
            if total_return > 5.0:
                print(f"  ⚠️  WARNING: Suspicious high return {total_return*100:.1f}%")
            if win_rate > 0.80:
                print(f"  ⚠️  WARNING: Unrealistic win rate {win_rate*100:.1f}%")
            if wait_rate > 0.70:
                print(f"  ⚠️  WARNING: High WAIT rate {wait_rate*100:.1f}%")
                print(f"     Consider lowering consensus threshold or checking agent calibration")
            
            # Sample trades
            print(f"\n  📝 Sample trades:")
            for i, trade in enumerate(trades[:5], 1):
                print(f"    {i}. {trade['entry_date']} → {trade['exit_date']}")
                print(f"       ${trade['entry_price']} → ${trade['exit_price']} | {trade['action']} {trade['shares']} shares")
                print(f"       Hit: {trade['hit_level']} | P/L: {trade['profit_pct']*100:.2f}%")
            
            suspicious = [t for t in trades if abs(t["profit_pct"]) > 0.5]
            if suspicious:
                print(f"\n  ⚠️  SUSPICIOUS TRADES (>50%):")
                for trade in suspicious[:3]:
                    print(f"    {trade['entry_date']}: ${trade['entry_price']} → ${trade['exit_price']}, P/L={trade['profit_pct']*100:.1f}%")
        else:
            print(f"  ❌ Нет сделок")
            wait_rate = ticker_signal_stats["wait"] / max(1, ticker_signal_stats["checks"])
            print(f"     WAIT Rate: {wait_rate*100:.1f}% ({ticker_signal_stats['wait']}/{ticker_signal_stats['checks']} checks)")
            if wait_rate > 0.90:
                print(f"  ⚠️  Octopus returning WAIT >90% of the time for {ticker}")
    
    # Сохранение
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"octopus_backtest_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "backtest_date": datetime.now().isoformat(),
            "version": "7.0_production_perfection",
            "orchestrator": "analyze_asset_octopus",
            "agents": ["Global", "M7", "W7", "AlphaPulse"],
            "weights": {"Global": 0.13, "M7": 0.20, "W7": 0.26, "AlphaPulse": 0.28},
            "initial_capital": INITIAL_CAPITAL,
            "risk_per_trade_pct": RISK_PER_TRADE_PCT,
            "period_days": LOOKBACK_DAYS,
            "commission_pct": COMMISSION_PCT,
            "slippage_pct": SLIPPAGE_PCT,
            "backtest_mode": True,
            "ml_models_available": ml_available,
            "signal_statistics": signal_stats,
            "tickers": TICKERS,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"✅ Backtest завершен!")
    print(f"{'='*60}")
    print(f"📁 Результаты: {output_file}\n")
    
    if all_results:
        print("📈 Общая статистика:")
        print(f"   Всего сделок: {sum([r['total_trades'] for r in all_results])}")
        print(f"   Средний Sharpe: {np.mean([r['sharpe_ratio'] for r in all_results]):.2f}")
        print(f"   Средний Win Rate: {np.mean([r['win_rate_pct'] for r in all_results]):.1f}%")
        
        overall_wait_rate = signal_stats["wait"] / max(1, signal_stats["total_checks"])
        print(f"   Overall WAIT Rate: {overall_wait_rate*100:.1f}%")
        print(f"   Signal Distribution: BUY={signal_stats['buy']}, SHORT={signal_stats['short']}, WAIT={signal_stats['wait']}")


if __name__ == "__main__":
    run_octopus_backtest()
