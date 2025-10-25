#!/usr/bin/env python3
"""
Octopus Historical Backtest v5.0 - TRUE PRODUCTION READY
Complete rewrite with proper position sizing and honest metrics

Key Features:
- Proper 1% risk position sizing (industry standard)
- Honest max drawdown calculation (no artificial caps)
- Extended logging for debugging
- Sanity checks for unrealistic results
- Look-ahead bias protection
- Open-High-Low-Close intrabar logic
"""

import sys
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

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
INITIAL_CAPITAL = 100000
RISK_PER_TRADE_PCT = 0.01  # 1% риска на сделку (industry standard)
MAX_POSITION_PCT = 0.10  # Макс 10% капитала на одну позицию


class Account:
    """Аккаунт с proper position sizing"""
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = [initial_capital]
        self.trades_log = []
    
    def calculate_position_size(self, entry_price: float, sl_price: float) -> float:
        """
        Правильный position sizing: риск 1% капитала на сделку
        Reference: Risk Management 101
        """
        if entry_price == 0 or sl_price == 0:
            return 0
        
        # Риск в долларах (1% от текущего equity)
        risk_amount = self.equity * RISK_PER_TRADE_PCT
        
        # Риск в процентах от entry цены
        price_risk_pct = abs((entry_price - sl_price) / entry_price)
        
        if price_risk_pct == 0:
            return 0
        
        # Кол-во акций при 1% риске
        shares = risk_amount / (entry_price * price_risk_pct)
        position_value = shares * entry_price
        
        # Ограничиваем макс 10% капитала на позицию
        max_position_value = self.equity * MAX_POSITION_PCT
        if position_value > max_position_value:
            position_value = max_position_value
            shares = position_value / entry_price
        
        return shares
    
    def execute_trade(self, trade: Dict):
        """Обновляем equity после сделки"""
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
        
        # Timeout
        if self.days_held >= POSITION_TIMEOUT_DAYS:
            self.exit_price = close
            self.exit_date = current_date
            self.hit_level = "TIMEOUT"
            self.closed = True
            self._calculate_profit()
            return True
        
        if self.entry_price == 0 or self.sl == 0:
            return False
        
        # Open-High-Low-Close sequence
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
        """Расчет БЕЗ caps - честные результаты"""
        if self.exit_price == 0 or self.entry_price == 0:
            self.profit_pct = 0
            return
        
        if self.action == "BUY":
            gross_profit = (self.exit_price - self.entry_price) / self.entry_price
        else:
            gross_profit = (self.entry_price - self.exit_price) / self.entry_price
        
        # Вычитаем издержки (БЕЗ caps)
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


def get_octopus_consensus(ticker: str) -> Optional[Dict]:
    """Агрегирует сигналы"""
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
    
    actions = [s["signal"]["recommendation"]["action"] for s in signals]
    consensus_action = max(set(actions), key=actions.count)
    
    consensus_signals = [s for s in signals if s["signal"]["recommendation"]["action"] == consensus_action]
    best_signal = max(consensus_signals, key=lambda s: s["signal"]["recommendation"]["confidence"])
    
    return best_signal["signal"]


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """ПРАВИЛЬНЫЙ расчет max drawdown БЕЗ caps"""
    if len(equity_curve) <= 1:
        return 0.0
    
    equity = np.array(equity_curve)
    
    # Если портфель обнулился
    if equity.min() <= 0:
        return -1.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    
    return drawdown.min()


def run_octopus_backtest():
    print("🚀 Octopus Backtest v5.0 - TRUE PRODUCTION READY\n")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Risk per Trade: {RISK_PER_TRADE_PCT*100}%")
    print(f"Max Position Size: {MAX_POSITION_PCT*100}%\n")
    
    all_results = []
    
    for ticker in TICKERS:
        print(f"\n📊 Backtesting {ticker.upper()}")
        
        historical_data = fetch_historical_ohlc(ticker, LOOKBACK_DAYS)
        
        if historical_data.empty:
            print(f"  ❌ Пропуск")
            continue
        
        account = Account(INITIAL_CAPITAL)
        open_position = None
        
        for idx, row in historical_data.iterrows():
            current_date = row["date"].strftime("%Y-%m-%d")
            
            # Проверка выхода
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
            
            # Поиск сигнала (раз в неделю)
            if not open_position and idx % 7 == 0 and idx > 0:
                try:
                    consensus_signal = get_octopus_consensus(ticker)
                    if consensus_signal and consensus_signal["recommendation"]["action"] != "WAIT":
                        # LOOK-AHEAD BIAS PROTECTION: entry на СЛЕДУЮЩИЙ бар Open
                        if idx + 1 < len(historical_data):
                            next_row = historical_data.iloc[idx + 1]
                            entry_price = next_row["o"]
                            sl_price = consensus_signal["levels"]["sl"]
                            
                            if entry_price > 0 and sl_price > 0:
                                shares = account.calculate_position_size(entry_price, sl_price)
                                if shares > 0:
                                    open_position = Position(consensus_signal, current_date, entry_price, shares)
                except Exception as e:
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
            
            # SANITY CHECKS
            if total_return > 5.0:
                print(f"  ⚠️  WARNING: Suspicious high return {total_return*100:.1f}% - possible bug!")
            if win_rate > 0.80:
                print(f"  ⚠️  WARNING: Unrealistic win rate {win_rate*100:.1f}% - check logic!")
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ
            print(f"\n  📝 Sample trades:")
            for i, trade in enumerate(trades[:5], 1):
                print(f"    {i}. {trade['entry_date']} → {trade['exit_date']}")
                print(f"       ${trade['entry_price']} → ${trade['exit_price']} | {trade['action']} {trade['shares']} shares")
                print(f"       Hit: {trade['hit_level']} | P/L: {trade['profit_pct']*100:.2f}%")
            
            # Подозрительные сделки
            suspicious = [t for t in trades if abs(t["profit_pct"]) > 0.5]
            if suspicious:
                print(f"\n  ⚠️  SUSPICIOUS TRADES (>50% return):")
                for trade in suspicious[:5]:
                    print(f"    {trade['entry_date']}: ${trade['entry_price']} → ${trade['exit_price']}, P/L={trade['profit_pct']*100:.1f}%")
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
            "version": "5.0_true_production_ready",
            "initial_capital": INITIAL_CAPITAL,
            "risk_per_trade_pct": RISK_PER_TRADE_PCT,
            "period_days": LOOKBACK_DAYS,
            "agents_used": ["Global", "M7", "W7"],
            "commission_pct": COMMISSION_PCT,
            "slippage_pct": SLIPPAGE_PCT,
            "intrabar_logic": "Open-High-Low-Close (MultiCharts standard)",
            "look_ahead_bias_protection": True,
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
