#!/usr/bin/env python3
"""
Octopus Historical Backtest v9.1 - IMPORT FIX
Fixed PolygonClient import issue + all v9.0 improvements
"""

import sys
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# === BACKTEST MODE FLAG ===
os.environ["BACKTEST_MODE"] = "1"
print("🔧 Backtest mode enabled (performance logging disabled)\n")

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY not set")

print(f"Using API key: {POLYGON_API_KEY[:8]}***\n")

# === SAFE IMPORTS ===
print("📦 Importing dependencies...")

import pandas as pd
import numpy as np

# Try to import PolygonClient from multiple locations
PolygonClient = None
try:
    from core.data_client import PolygonClient as PC
    PolygonClient = PC
    print("  ✅ PolygonClient imported from core.data_client")
except ImportError:
    try:
        from core.strategy import PolygonClient as PC
        PolygonClient = PC
        print("  ✅ PolygonClient imported from core.strategy")
    except ImportError:
        print("  ⚠️  PolygonClient not found, creating simple version")

# Create simple PolygonClient if not found
if PolygonClient is None:
    class PolygonClient:
        """Simple Polygon API client"""
        def __init__(self):
            self.api_key = POLYGON_API_KEY
        
        def daily_ohlc(self, ticker: str, days: int = 120) -> pd.DataFrame:
            """Fetch OHLC data from Polygon"""
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apiKey": self.api_key, "limit": 5000}
            
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if "results" not in data or not data["results"]:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data["results"])
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
                df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                return df[["timestamp", "open", "high", "low", "close", "volume"]]
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                return pd.DataFrame()
    
    print("  ✅ Simple PolygonClient created")

# Import agents with fallbacks
try:
    from core.strategy import (
        analyze_asset_octopus,
        analyze_asset_global,
        analyze_asset_w7,
        _atr_like,
        _clip01,
        _monotone_tp_probs,
        logger
    )
    print("  ✅ Core strategy functions imported")
except ImportError as e:
    print(f"  ❌ Failed to import core functions: {e}")
    raise

# Safe import AlphaPulse
try:
    from core.strategy import analyze_asset_alphapulse
    print("  ✅ AlphaPulse agent imported")
except ImportError:
    print("  ⚠️  AlphaPulse agent not found (will use 3 agents)")
    analyze_asset_alphapulse = None

# Safe import CAL_CONF
try:
    from core.strategy import CAL_CONF
    print("  ✅ CAL_CONF imported")
except ImportError:
    print("  ⚠️  CAL_CONF not found, using identity calibration")
    CAL_CONF = {}

print()

# === M7 BACKTEST MONKEY PATCH ===
print("🔧 Applying M7 monkey patch...")
print("""
⚠️  IMPORTANT: M7 Strategy Modified for Backtest
   Original M7: Pivot points + Fibonacci levels + limit orders
   Backtest M7: Simple MA(10/30) crossover + ATR levels + market orders
   
   This change affects:
   - Signal timing (different entry points)
   - Octopus voting results (M7 votes differently)
   - Overall performance (simpler strategy)
   
   Reason: Original M7 uses limit orders at key levels, which don't work
   in backtest with market orders. This is expected behavior.
""")

def analyze_asset_m7_backtest(ticker, horizon="Краткосрочный", use_ml=False):
    """M7 BACKTEST-COMPATIBLE VERSION (monkey patch)"""
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    
    if df is None or df.empty:
        return {
            "last_price": 0.0,
            "recommendation": {"action":"WAIT","confidence":0.5},
            "levels":{"entry":0,"sl":0,"tp1":0,"tp2":0,"tp3":0},
            "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "context":["Нет данных для M7"],
            "note_html":"<div>M7: ожидание</div>",
            "alt":"Ожидание",
            "entry_kind":"wait",
            "entry_label":"WAIT",
            "meta":{"source":"M7","grey_zone":True,"backtest_mode":True}
        }

    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    
    # Get price from correct column name
    if 'close' in df.columns:
        price = float(df['close'].iloc[-1])
        closes = df['close']
    elif 'c' in df.columns:
        price = float(df['c'].iloc[-1])
        closes = df['c']
    else:
        return {"last_price": 0.0, "recommendation": {"action":"WAIT","confidence":0.5},
                "levels":{"entry":0,"sl":0,"tp1":0,"tp2":0,"tp3":0},
                "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0}, "context":["No price data"],
                "note_html":"<div>M7: no data</div>", "alt":"No data", "entry_kind":"wait",
                "entry_label":"WAIT", "meta":{"source":"M7","backtest_mode":True}}
    
    atr14 = float(_atr_like(df, n=14).iloc[-1]) or 1e-9
    
    # Simple MA crossover strategy
    if len(closes) >= 30:
        short_ma = closes.rolling(10).mean().iloc[-1]
        long_ma = closes.rolling(30).mean().iloc[-1]
        ma_gap = float((short_ma - long_ma) / max(1e-9, long_ma))
    else:
        ma_gap = 0.0
    
    action = "BUY" if ma_gap > 0 else "SHORT"
    
    # Base confidence
    base_conf = 0.58 + 0.20*min(1.0, abs(ma_gap)/0.02)
    confidence = max(0.52, min(0.78, base_conf))
    
    # Apply calibration safely
    try:
        m7_calibrator = CAL_CONF.get("M7", lambda x: x)
        confidence = float(m7_calibrator(confidence))
    except (AttributeError, KeyError, TypeError):
        confidence = max(0.52, min(0.82, confidence))
    
    # ATR-based levels
    entry = price
    if action == "BUY":
        sl = price - 2.0*atr14
        tp1, tp2, tp3 = price + 1.5*atr14, price + 2.5*atr14, price + 4.0*atr14
    else:
        sl = price + 2.0*atr14
        tp1, tp2, tp3 = price - 1.5*atr14, price - 2.5*atr14, price - 4.0*atr14
    
    # Probabilities
    u1, u2, u3 = 1.5, 2.5, 4.0
    k = 0.18
    p1 = _clip01(confidence * np.exp(-k*(u1-1.0)))
    p2 = _clip01(max(0.50, confidence-0.08) * np.exp(-k*(u2-1.5)))
    p3 = _clip01(max(0.45, confidence-0.16) * np.exp(-k*(u3-2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})
    
    return {
        "last_price": price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [f"M7(backtest): MA gap {ma_gap:.2%}"],
        "note_html": f"<div>M7: {action} с {confidence:.0%}</div>",
        "alt": "M7 backtest mode",
        "entry_kind": "market",
        "entry_label": f"{action} NOW",
        "meta": {
            "source": "M7",
            "backtest_mode": True,
            "ma_gap": float(ma_gap),
            "strategy_type": "ma_crossover",
            "original_strategy": "pivot_points"
        }
    }

# Apply monkey patch
import core.strategy
core.strategy.analyze_asset_m7 = analyze_asset_m7_backtest

# Verify patch worked
try:
    test_result = core.strategy.analyze_asset_m7("X:BTCUSD", "Краткосрочный")
    if test_result.get("meta", {}).get("backtest_mode") == True:
        print("✅ M7 monkey patch verified successfully\n")
    else:
        print("⚠️  WARNING: M7 monkey patch may not be applied correctly\n")
except Exception as e:
    print(f"⚠️  WARNING: M7 patch verification failed: {e}\n")

# === CONFIGURATION ===
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
    
    expected_models = [
        models_dir / "arxora_m7pro" / f"{ticker.replace(':', '_')}_model.joblib"
        for ticker in TICKERS
    ]
    
    missing = [m for m in expected_models if not m.exists()]
    
    if missing:
        print(f"  ⚠️  Missing {len(missing)}/{len(expected_models)} ML models")
        print(f"  ℹ️  M7 agent will run without ML\n")
        return False
    else:
        print(f"  ✅ All ML models found ({len(expected_models)} models)\n")
        return True


class Account:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = [initial_capital]
        self.trades_log = []
    
    def calculate_position_size(self, entry_price: float, sl_price: float) -> float:
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
        profit_loss = trade["profit_pct"] * trade["position_value"]
        self.equity += profit_loss
        self.equity_curve.append(self.equity)
        self.trades_log.append(trade)


class Position:
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
        self.profit_dollars = 0
        self.hit_level = None
        self.days_held = 0
        
        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {self.entry_price}")

    def check_exit(self, bar: pd.Series, current_date: str) -> bool:
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
        """FIXED: Correct profit calculation for SHORT positions"""
        if self.exit_price == 0 or self.entry_price == 0 or self.position_value == 0:
            self.profit_pct = 0
            self.profit_dollars = 0
            return
        
        if self.action == "BUY":
            price_change = self.exit_price - self.entry_price
        else:  # SHORT
            price_change = self.entry_price - self.exit_price
        
        gross_profit_dollars = price_change * self.shares
        total_costs = self.position_value * (COMMISSION_PCT + SLIPPAGE_PCT) * 2
        net_profit_dollars = gross_profit_dollars - total_costs
        
        self.profit_pct = net_profit_dollars / self.position_value
        self.profit_dollars = net_profit_dollars
        
        if self.profit_pct < -0.95:
            print(f"  ⚠️  Capping extreme loss: {self.profit_pct*100:.1f}% → -95%")
            self.profit_pct = -0.95


def fetch_historical_ohlc(ticker: str, days: int = 730) -> pd.DataFrame:
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
    if len(equity_curve) <= 1:
        return 0.0
    
    equity = np.array(equity_curve)
    
    if equity.min() <= 0:
        return -1.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    
    return drawdown.min()


def run_octopus_backtest():
    print("🚀 Octopus Backtest v9.1 - IMPORT FIX + PRODUCTION READY\n")
    
    ml_available = check_ml_models()
    
    print(f"""
📝 Octopus Configuration:
   Orchestrator: analyze_asset_octopus()
   Agents: Global (0.13), M7 (0.20 - PATCHED), W7 (0.26), AlphaPulse (0.28)
   
   Risk Management:
   - Initial Capital: ${INITIAL_CAPITAL:,}
   - Risk per Trade: {RISK_PER_TRADE_PCT*100}%
   - Max Position: {MAX_POSITION_PCT*100}%
   - Position Timeout: {POSITION_TIMEOUT_DAYS} days
""")
    
    all_results = []
    signal_stats = {"total_checks": 0, "wait": 0, "buy": 0, "short": 0}
    agent_errors = {}
    
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
        ticker_agent_errors = []
        
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
                        "shares": round(open_position.shares, 4),
                        "position_value": round(open_position.position_value, 2),
                        "hit_level": open_position.hit_level,
                        "profit_pct": round(open_position.profit_pct, 4),
                        "profit_dollars": round(open_position.profit_dollars, 2),
                        "days_held": open_position.days_held
                    }
                    account.execute_trade(trade)
                    open_position = None
            
            if not open_position and idx % 7 == 0 and idx > 0:
                ticker_signal_stats["checks"] += 1
                signal_stats["total_checks"] += 1
                
                try:
                    octopus_signal = analyze_asset_octopus(ticker, "Краткосрочный")
                    action = octopus_signal["recommendation"]["action"]
                    
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
                    error_msg = str(e)[:100]
                    ticker_agent_errors.append({"date": current_date, "error": error_msg})
                    if len(ticker_agent_errors) <= 3:
                        print(f"  ⚠️  Error at {current_date}: {error_msg}")
                    continue
        
        if ticker_agent_errors:
            agent_errors[ticker] = ticker_agent_errors
        
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
                "shares": round(open_position.shares, 4),
                "position_value": round(open_position.position_value, 2),
                "hit_level": "FORCED_CLOSE",
                "profit_pct": round(open_position.profit_pct, 4),
                "profit_dollars": round(open_position.profit_dollars, 2),
                "days_held": open_position.days_held
            }
            account.execute_trade(trade)
        
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
                "agent_errors_count": len(ticker_agent_errors),
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
            print(f"     WAIT Rate: {result['wait_rate_pct']}%")
            
            if total_return > 5.0:
                print(f"  ⚠️  WARNING: Suspicious high return {total_return*100:.1f}%")
            if win_rate > 0.80:
                print(f"  ⚠️  WARNING: Unrealistic win rate {win_rate*100:.1f}%")
            
            print(f"\n  📝 Sample trades:")
            for i, trade in enumerate(trades[:5], 1):
                print(f"    {i}. {trade['entry_date']} → {trade['exit_date']}")
                print(f"       ${trade['entry_price']} → ${trade['exit_price']} | {trade['action']}")
                print(f"       Hit: {trade['hit_level']} | P/L: ${trade['profit_dollars']:,.2f} ({trade['profit_pct']*100:.2f}%)")
        else:
            print(f"  ❌ Нет сделок")
            wait_rate = ticker_signal_stats["wait"] / max(1, ticker_signal_stats["checks"])
            print(f"     WAIT Rate: {wait_rate*100:.1f}% ({ticker_signal_stats['wait']}/{ticker_signal_stats['checks']} checks)")
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"octopus_backtest_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "backtest_date": datetime.now().isoformat(),
            "version": "9.1_import_fix",
            "orchestrator": "analyze_asset_octopus",
            "agents": ["Global", "M7 (patched)", "W7", "AlphaPulse"],
            "m7_patch_applied": True,
            "m7_strategy_changed": "pivot_points -> ma_crossover",
            "initial_capital": INITIAL_CAPITAL,
            "risk_per_trade_pct": RISK_PER_TRADE_PCT,
            "commission_pct": COMMISSION_PCT,
            "slippage_pct": SLIPPAGE_PCT,
            "tickers": TICKERS,
            "signal_statistics": signal_stats,
            "agent_errors": agent_errors,
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
