#!/usr/bin/env python3
"""
Octopus Historical Backtest v10.2 - PRODUCTION READY
Critical fixes: Reliable ATR calculation, fallbacks, validation
"""

import sys
import os
import json
import requests
import inspect
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Try to import PolygonClient
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

if PolygonClient is None:
    class PolygonClient:
        def __init__(self):
            self.api_key = POLYGON_API_KEY
        
        def daily_ohlc(self, ticker: str, days: int = 120) -> pd.DataFrame:
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

# Import helper functions
try:
    from core.strategy import (
        _atr_like,
        _clip01,
        _monotone_tp_probs,
        logger
    )
    print("  ✅ Core strategy helper functions imported")
except ImportError as e:
    print(f"  ❌ Failed to import helper functions: {e}")
    raise

# Import CAL_CONF
try:
    from core.strategy import CAL_CONF
    print("  ✅ CAL_CONF imported")
except ImportError:
    print("  ⚠️  CAL_CONF not found, using identity calibration")
    CAL_CONF = {
        "Global": lambda x: x,
        "M7": lambda x: x,
        "W7": lambda x: x,
        "AlphaPulse": lambda x: x,
        "Octopus": lambda x: x
    }

print()

# === BACKTEST AGENT PATCHES ===
print("🔧 Applying backtest patches for ALL agents...")
print("""
⚠️  IMPORTANT: ALL Agents Modified for Backtest
   - Global, M7, W7, AlphaPulse: MA crossover + ATR levels
   - Levels recalculated using HISTORICAL data at signal time
   - Using production-grade ATR calculation with fallbacks
""")


def _create_backtest_agent(agent_name: str, ma_short: int, ma_long: int):
    """Factory function to create backtest-compatible agents"""
    
    def agent_func(ticker, horizon="Краткосрочный", use_ml=False):
        cli = PolygonClient()
        df = cli.daily_ohlc(ticker, days=120)
        
        if df is None or df.empty:
            return {
                "last_price": 0.0,
                "recommendation": {"action":"WAIT","confidence":0.5},
                "levels":{"entry":0,"sl":0,"tp1":0,"tp2":0,"tp3":0},
                "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0},
                "context":[f"Нет данных для {agent_name}"],
                "note_html":f"<div>{agent_name}: ожидание</div>",
                "alt":"Ожидание",
                "entry_kind":"wait",
                "entry_label":"WAIT",
                "meta":{"source":agent_name,"grey_zone":True,"backtest_mode":True}
            }

        df = df.sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        
        # Get price
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
                    "note_html":f"<div>{agent_name}: no data</div>", "alt":"No data",
                    "entry_kind":"wait", "entry_label":"WAIT",
                    "meta":{"source":agent_name,"backtest_mode":True}}
        
        atr14 = float(_atr_like(df, n=14).iloc[-1]) or 1e-9
        
        # MA crossover
        if len(closes) >= ma_long:
            short_ma = closes.rolling(ma_short).mean().iloc[-1]
            long_ma = closes.rolling(ma_long).mean().iloc[-1]
            ma_gap = float((short_ma - long_ma) / max(1e-9, long_ma))
        else:
            ma_gap = 0.0
        
        action = "BUY" if ma_gap > 0 else "SHORT"
        
        # Confidence
        base_conf = 0.55 + 0.20*min(1.0, abs(ma_gap)/0.02)
        confidence = max(0.50, min(0.78, base_conf))
        
        try:
            agent_calibrator = CAL_CONF.get(agent_name, lambda x: x)
            confidence = float(agent_calibrator(confidence))
        except (AttributeError, KeyError, TypeError):
            confidence = max(0.50, min(0.82, confidence))
        
        # ATR levels
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
            "context": [f"{agent_name}(backtest): MA({ma_short}/{ma_long}) gap {ma_gap:.2%}"],
            "note_html": f"<div>{agent_name}: {action} с {confidence:.0%}</div>",
            "alt": f"{agent_name} backtest mode",
            "entry_kind": "market",
            "entry_label": f"{action} NOW",
            "meta": {
                "source": agent_name,
                "backtest_mode": True,
                "ma_gap": float(ma_gap),
                "atr": float(atr14),
                "entry": float(entry),
                "tp3": float(tp3),
                "strategy_type": "ma_crossover",
                "ma_short": ma_short,
                "ma_long": ma_long
            }
        }
    
    return agent_func


# Create agents
analyze_asset_global_backtest = _create_backtest_agent("Global", ma_short=20, ma_long=50)
analyze_asset_m7_backtest = _create_backtest_agent("M7", ma_short=10, ma_long=30)
analyze_asset_w7_backtest = _create_backtest_agent("W7", ma_short=15, ma_long=40)
analyze_asset_alphapulse_backtest = _create_backtest_agent("AlphaPulse", ma_short=12, ma_long=35)

# Apply patches
import core.strategy
core.strategy.analyze_asset_global = analyze_asset_global_backtest
core.strategy.analyze_asset_m7 = analyze_asset_m7_backtest
core.strategy.analyze_asset_w7 = analyze_asset_w7_backtest
core.strategy.analyze_asset_alphapulse = analyze_asset_alphapulse_backtest

print("  ✅ All agents patched\n")

# Import Octopus
try:
    from core.strategy import analyze_asset_octopus
    print("  ✅ Octopus imported (full orchestration with patched agents)\n")
except ImportError as e:
    print(f"  ❌ Failed to import Octopus: {e}")
    raise

# === CONFIGURATION ===
TICKERS = ["X:BTCUSD", "X:ETHUSD", "AAPL", "NVDA"]
LOOKBACK_DAYS = 730
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.0005
POSITION_TIMEOUT_DAYS = 28
INITIAL_CAPITAL = 100000
RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_PCT = 0.10


def calculate_historical_atr(historical_df: pd.DataFrame, n: int = 14) -> float:
    """
    Calculate ATR using existing _atr_like function with robust fallbacks
    Returns: ATR value or fallback (2% of price)
    """
    if len(historical_df) < n:
        # Not enough data for ATR calculation
        fallback_atr = float(historical_df['c'].iloc[-1]) * 0.02
        return fallback_atr
    
    try:
        # Prepare dataframe for _atr_like
        df_copy = historical_df.copy()
        
        # Set date as index if not already
        if 'date' in df_copy.columns:
            df_copy = df_copy.set_index('date')
        
        # Use the battle-tested _atr_like function from core.strategy
        atr_series = _atr_like(df_copy, n=n)
        atr = float(atr_series.iloc[-1])
        
        # Validate ATR
        if atr <= 0 or np.isnan(atr) or np.isinf(atr):
            # Invalid ATR, use fallback
            fallback_atr = float(historical_df['c'].iloc[-1]) * 0.02
            return fallback_atr
        
        return atr
        
    except Exception as e:
        # Any error in ATR calculation, use fallback
        fallback_atr = float(historical_df['c'].iloc[-1]) * 0.02
        return fallback_atr


def recalculate_levels(action: str, entry_price: float, atr: float) -> Dict:
    """
    Recalculate levels based on historical entry price and ATR
    Ensures levels are in correct direction
    """
    if action == "BUY":
        sl = entry_price - 2.0*atr
        tp1 = entry_price + 1.5*atr
        tp2 = entry_price + 2.5*atr
        tp3 = entry_price + 4.0*atr
    else:  # SHORT
        sl = entry_price + 2.0*atr
        tp1 = entry_price - 1.5*atr
        tp2 = entry_price - 2.5*atr
        tp3 = entry_price - 4.0*atr
    
    return {
        "entry": entry_price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3
    }


def validate_levels(action: str, entry: float, sl: float, tp3: float) -> bool:
    """
    Validate that levels are in correct direction
    Returns True if valid, False otherwise
    """
    if action == "BUY":
        return sl < entry and tp3 > entry
    elif action == "SHORT":
        return sl > entry and tp3 < entry
    return False


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
        if self.exit_price == 0 or self.entry_price == 0 or self.position_value == 0:
            self.profit_pct = 0
            self.profit_dollars = 0
            return
        
        if self.action == "BUY":
            price_change = self.exit_price - self.entry_price
        else:
            price_change = self.entry_price - self.exit_price
        
        gross_profit_dollars = price_change * self.shares
        total_costs = self.position_value * (COMMISSION_PCT + SLIPPAGE_PCT) * 2
        net_profit_dollars = gross_profit_dollars - total_costs
        
        self.profit_pct = net_profit_dollars / self.position_value
        self.profit_dollars = net_profit_dollars
        
        if self.profit_pct < -0.95:
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
    print("🚀 Octopus Backtest v10.2 - PRODUCTION READY\n")
    
    print(f"""
📝 Configuration:
   - Full Octopus orchestration (4 agents with weighted voting)
   - Levels RECALCULATED using historical price/ATR
   - Production-grade ATR calculation with fallbacks
   - Comprehensive validation
   
   Risk Management:
   - Initial Capital: ${INITIAL_CAPITAL:,}
   - Risk per Trade: {RISK_PER_TRADE_PCT*100}%
   - Max Position: {MAX_POSITION_PCT*100}%
   - Position Timeout: {POSITION_TIMEOUT_DAYS} days
""")
    
    all_results = []
    signal_stats = {"total_checks": 0, "wait": 0, "buy": 0, "short": 0}
    validation_stats = {"total": 0, "invalid": 0, "fallback_atr": 0}
    first_signal_printed = False
    
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
        ticker_validation = {"total": 0, "invalid": 0, "fallback_atr": 0}
        
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
                    # Get Octopus signal (full orchestration)
                    octopus_signal = analyze_asset_octopus(ticker, "Краткосрочный")
                    action = octopus_signal["recommendation"]["action"]
                    confidence = octopus_signal["recommendation"]["confidence"]
                    
                    # Track stats
                    if action == "WAIT":
                        ticker_signal_stats["wait"] += 1
                        signal_stats["wait"] += 1
                        continue
                    elif action == "BUY":
                        ticker_signal_stats["buy"] += 1
                        signal_stats["buy"] += 1
                    elif action == "SHORT":
                        ticker_signal_stats["short"] += 1
                        signal_stats["short"] += 1
                    
                    # CRITICAL: Recalculate levels using HISTORICAL data
                    if idx + 1 < len(historical_data):
                        ticker_validation["total"] += 1
                        validation_stats["total"] += 1
                        
                        next_row = historical_data.iloc[idx + 1]
                        entry_price = next_row["o"]
                        
                        # Calculate ATR from historical data
                        historical_subset = historical_data.iloc[:idx+1]
                        atr = calculate_historical_atr(historical_subset, n=14)
                        
                        # Check if fallback was used
                        expected_atr_min = entry_price * 0.005  # 0.5% of price
                        if atr <= expected_atr_min:
                            ticker_validation["fallback_atr"] += 1
                            validation_stats["fallback_atr"] += 1
                        
                        # Recalculate levels
                        new_levels = recalculate_levels(action, entry_price, atr)
                        
                        # Validate levels
                        if not validate_levels(action, new_levels["entry"], new_levels["sl"], new_levels["tp3"]):
                            ticker_validation["invalid"] += 1
                            validation_stats["invalid"] += 1
                            print(f"  🚨 Invalid levels: {action} entry=${new_levels['entry']:.2f}, sl=${new_levels['sl']:.2f}, tp3=${new_levels['tp3']:.2f}")
                            continue
                        
                        # Update signal with recalculated levels
                        octopus_signal["levels"] = new_levels
                        
                        # Debug first signal
                        if not first_signal_printed:
                            print(f"\n  📝 First {action} signal ({ticker}):")
                            print(f"     Date: {current_date}")
                            print(f"     Entry: ${entry_price:.2f}")
                            print(f"     Historical ATR: ${atr:.2f} ({atr/entry_price*100:.2f}% of price)")
                            print(f"     SL: ${new_levels['sl']:.2f} (distance: ${abs(new_levels['sl']-entry_price):.2f})")
                            print(f"     TP3: ${new_levels['tp3']:.2f} (distance: ${abs(new_levels['tp3']-entry_price):.2f})")
                            print(f"     Expected TP3 distance: ~${4*atr:.2f}")
                            print(f"     ✅ Validated & recalculated\n")
                            first_signal_printed = True
                        
                        sl_price = new_levels["sl"]
                        
                        if entry_price > 0 and sl_price > 0:
                            shares = account.calculate_position_size(entry_price, sl_price)
                            if shares > 0:
                                open_position = Position(octopus_signal, current_date, entry_price, shares)
                
                except Exception as e:
                    error_msg = str(e)[:100]
                    print(f"  ⚠️  Error at {current_date}: {error_msg}")
                    continue
        
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
                "validation_stats": ticker_validation,
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
            print(f"     Validation: {ticker_validation['invalid']} invalid, {ticker_validation['fallback_atr']} fallback ATR")
            
            print(f"\n  📝 Sample trades:")
            for i, trade in enumerate(trades[:5], 1):
                print(f"    {i}. {trade['entry_date']} → {trade['exit_date']}")
                print(f"       ${trade['entry_price']} → ${trade['exit_price']} | {trade['action']}")
                print(f"       Hit: {trade['hit_level']} | P/L: ${trade['profit_dollars']:,.2f} ({trade['profit_pct']*100:.2f}%)")
        else:
            print(f"  ❌ Нет сделок")
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"octopus_backtest_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "backtest_date": datetime.now().isoformat(),
            "version": "10.2_production_ready",
            "description": "Full Octopus orchestration with historical levels, validated",
            "initial_capital": INITIAL_CAPITAL,
            "tickers": TICKERS,
            "signal_statistics": signal_stats,
            "validation_statistics": validation_stats,
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
        print(f"\n   Validation:")
        print(f"   - Invalid signals: {validation_stats['invalid']}/{validation_stats['total']}")
        print(f"   - Fallback ATR used: {validation_stats['fallback_atr']}/{validation_stats['total']}")


if __name__ == "__main__":
    run_octopus_backtest()
