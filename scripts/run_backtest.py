#!/usr/bin/env python3
"""
Octopus Historical Backtest v12.2 - ALPHAPULSE FIX + MAJORITY VOTING
Fixed: AlphaPulse use_ml parameter error
"""

import sys
import os
import json
import requests
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

# === IMPORTS ===
print("📦 Importing dependencies...")

import pandas as pd
import numpy as np

try:
    from core.data_client import PolygonClient
    print("  ✅ PolygonClient imported from core.data_client")
except ImportError:
    try:
        from core.strategy import PolygonClient
        print("  ✅ PolygonClient imported from core.strategy")
    except ImportError:
        raise ImportError("PolygonClient not found!")

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

try:
    from core.strategy import CAL_CONF
    print("  ✅ CAL_CONF imported")
except ImportError:
    print("  ⚠️  CAL_CONF not found, using identity calibration")
    CAL_CONF = {
        "M7": lambda x: x,
        "W7": lambda x: x,
        "AlphaPulse": lambda x: x,
        "Octopus": lambda x: x
    }

print()

# === IMPORT AGENTS ===
print("🎯 Importing M7, W7, AlphaPulse")

try:
    from core.strategy import (
        analyze_asset_m7,
        analyze_asset_w7,
        analyze_asset_alphapulse
    )
    print("  ✅ M7, W7, AlphaPulse imported\n")
except ImportError as e:
    print(f"  ❌ Failed to import agents: {e}")
    raise

# === CONFIGURATION ===
USE_MAJORITY_VOTING = True
VOTING_DEBUG = True
MAX_DEBUG_PRINTS = 3

OCTO_WEIGHTS_3 = {
    "M7": 0.27,
    "W7": 0.35,
    "AlphaPulse": 0.38
}

TICKERS = ["X:BTCUSD", "X:ETHUSD", "AAPL", "NVDA"]
LOOKBACK_DAYS = 730
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.0005
POSITION_TIMEOUT_DAYS = 28
INITIAL_CAPITAL = 100000
RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_PCT = 0.10
LIMIT_ORDER_WAIT_DAYS = 7

debug_count = 0


def analyze_asset_octopus_3agents(ticker: str, horizon: str = "Краткосрочный", debug: bool = False):
    """
    Custom Octopus using M7, W7, AlphaPulse
    With majority voting and enhanced debug
    """
    global debug_count
    
    try:
        # Get signals from 3 agents
        m7_sig = analyze_asset_m7(ticker, horizon)
        w7_sig = analyze_asset_w7(ticker, horizon)
        
        # ✅ FIX: Remove use_ml parameter (not supported by AlphaPulse)
        alpha_sig = analyze_asset_alphapulse(ticker, horizon)
        
        agents = {
            "M7": m7_sig,
            "W7": w7_sig,
            "AlphaPulse": alpha_sig
        }
        
        # Collect actions and confidences
        actions = []
        for name, sig in agents.items():
            action = sig["recommendation"]["action"]
            confidence = sig["recommendation"]["confidence"]
            actions.append((action, confidence, name))
        
        # DEBUG: Print agent votes
        if debug and VOTING_DEBUG and debug_count < MAX_DEBUG_PRINTS:
            print(f"\n  🗳️  Agent votes ({ticker}):")
            for action, confidence, name in actions:
                print(f"     {name}: {action} ({confidence:.0%})")
            debug_count += 1
        
        # MAJORITY VOTING
        if USE_MAJORITY_VOTING:
            buy_votes = sum(1 for a, c, n in actions if a == "BUY")
            short_votes = sum(1 for a, c, n in actions if a == "SHORT")
            
            if buy_votes >= 2:
                final_action = "BUY"
            elif short_votes >= 2:
                final_action = "SHORT"
            else:
                final_action = "WAIT"
            
            aligned_agents = [
                (a, c, n) for a, c, n in actions 
                if a == final_action
            ]
            
            if aligned_agents:
                avg_conf = sum(c for a, c, n in aligned_agents) / len(aligned_agents)
                final_confidence = max(0.50, min(0.85, avg_conf))
            else:
                final_confidence = 0.50
            
            if debug and VOTING_DEBUG and debug_count <= MAX_DEBUG_PRINTS:
                print(f"     Majority: BUY={buy_votes}, SHORT={short_votes}")
                print(f"     ✅ Final: {final_action} ({final_confidence:.0%})")
        
        # WEIGHTED VOTING
        else:
            def _act_to_num(a: str) -> int:
                return 1 if a == "BUY" else (-1 if a == "SHORT" else 0)
            
            weighted_sum = sum(
                _act_to_num(action) * confidence * OCTO_WEIGHTS_3[name]
                for action, confidence, name in actions
            )
            
            threshold = 0.05
            
            if weighted_sum > threshold:
                final_action = "BUY"
            elif weighted_sum < -threshold:
                final_action = "SHORT"
            else:
                final_action = "WAIT"
            
            aligned_conf = sum(
                confidence * OCTO_WEIGHTS_3[name]
                for action, confidence, name in actions
                if _act_to_num(action) * weighted_sum > 0
            )
            
            final_confidence = max(0.50, min(0.85, aligned_conf * 1.1))
            
            if debug and VOTING_DEBUG and debug_count <= MAX_DEBUG_PRINTS:
                print(f"     Weighted sum: {weighted_sum:.3f} (threshold: ±{threshold})")
                print(f"     ✅ Final: {final_action} ({final_confidence:.0%})")
        
        # Use primary agent's levels
        primary_agent = max(actions, key=lambda x: x[1])[2]
        primary_sig = agents[primary_agent]
        
        return {
            "last_price": primary_sig["last_price"],
            "recommendation": {
                "action": final_action,
                "confidence": final_confidence
            },
            "levels": primary_sig["levels"],
            "probs": primary_sig["probs"],
            "entry_kind": primary_sig.get("entry_kind", "market"),
            "entry_label": primary_sig.get("entry_label", f"{final_action} NOW"),
            "context": [f"Octopus-3: {'Majority' if USE_MAJORITY_VOTING else 'Weighted'} voting"],
            "note_html": f"<div>Octopus-3: {final_action} с {final_confidence:.0%}</div>",
            "alt": "Octopus 3-agent ensemble",
            "meta": {
                "orchestrator": "Octopus-3",
                "agents": list(agents.keys()),
                "voting_method": "majority" if USE_MAJORITY_VOTING else "weighted",
                "primary_agent": primary_agent,
                "votes": [(name, action, conf) for action, conf, name in actions]
            }
        }
    
    except Exception as e:
        print(f"  ⚠️  Octopus-3 error: {str(e)[:100]}")
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "context": [f"Error: {str(e)[:100]}"],
            "note_html": "<div>Error in Octopus-3</div>",
            "alt": "Error",
            "meta": {"error": str(e)}
        }


print(f"  ✅ Custom Octopus-3 created")
print(f"  ⚙️  Voting method: {'MAJORITY' if USE_MAJORITY_VOTING else 'WEIGHTED (threshold=0.05)'}\n")


def calculate_historical_atr(historical_df: pd.DataFrame, n: int = 14) -> float:
    if len(historical_df) < n:
        fallback_atr = float(historical_df['c'].iloc[-1]) * 0.02
        return fallback_atr
    
    try:
        df_copy = historical_df.copy()
        if 'date' in df_copy.columns:
            df_copy = df_copy.set_index('date')
        
        atr_series = _atr_like(df_copy, n=n)
        atr = float(atr_series.iloc[-1])
        
        if atr <= 0 or np.isnan(atr) or np.isinf(atr):
            fallback_atr = float(historical_df['c'].iloc[-1]) * 0.02
            return fallback_atr
        
        return atr
        
    except Exception as e:
        fallback_atr = float(historical_df['c'].iloc[-1]) * 0.02
        return fallback_atr


def recalculate_levels(action: str, entry_price: float, atr: float) -> Dict:
    if action == "BUY":
        sl = entry_price - 2.0*atr
        tp1 = entry_price + 1.5*atr
        tp2 = entry_price + 2.5*atr
        tp3 = entry_price + 4.0*atr
    else:
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
    if action == "BUY":
        return sl < entry and tp3 > entry
    elif action == "SHORT":
        return sl > entry and tp3 < entry
    return False


def check_limit_fill(bar: pd.Series, limit_price: float, action: str) -> bool:
    high = bar["h"]
    low = bar["l"]
    
    if action in ["BUY", "SHORT"]:
        return low <= limit_price <= high
    
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
    global debug_count
    
    print("🚀 Octopus Backtest v12.2 - ALPHAPULSE FIX + MAJORITY VOTING\n")
    
    print(f"""
📝 Configuration:
   - Agents: M7 (27%), W7 (35%), AlphaPulse (38%)
   - Voting: {'MAJORITY (2/3 consensus)' if USE_MAJORITY_VOTING else 'WEIGHTED (threshold=0.05)'}
   - Limit order simulation with {LIMIT_ORDER_WAIT_DAYS}-day wait window
   - Levels RECALCULATED using historical price/ATR
   - DEBUG: Shows agent votes for first {MAX_DEBUG_PRINTS} signals per ticker
   
   ✅ FIX: AlphaPulse use_ml parameter removed
   
   Risk Management:
   - Initial Capital: ${INITIAL_CAPITAL:,}
   - Risk per Trade: {RISK_PER_TRADE_PCT*100}%
   - Max Position: {MAX_POSITION_PCT*100}%
   - Position Timeout: {POSITION_TIMEOUT_DAYS} days
""")
    
    all_results = []
    signal_stats = {"total_checks": 0, "wait": 0, "buy": 0, "short": 0}
    limit_stats = {"total_limits": 0, "filled": 0, "expired": 0, "avg_wait_days": []}
    validation_stats = {"total": 0, "invalid": 0, "fallback_atr": 0}
    
    for ticker in TICKERS:
        debug_count = 0
        
        print(f"\n{'='*60}")
        print(f"📊 Backtesting {ticker.upper()}")
        print(f"{'='*60}")
        
        historical_data = fetch_historical_ohlc(ticker, LOOKBACK_DAYS)
        
        if historical_data.empty:
            print(f"  ❌ Пропуск")
            continue
        
        account = Account(INITIAL_CAPITAL)
        open_position = None
        pending_limit = None
        ticker_signal_stats = {"checks": 0, "wait": 0, "buy": 0, "short": 0}
        ticker_limit_stats = {"total": 0, "filled": 0, "expired": 0, "wait_days": []}
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
            
            if pending_limit and not open_position:
                limit_data = pending_limit
                days_waiting = (pd.to_datetime(current_date) - pd.to_datetime(limit_data["order_date"])).days
                
                if check_limit_fill(row, limit_data["limit_price"], limit_data["action"]):
                    ticker_limit_stats["filled"] += 1
                    ticker_limit_stats["wait_days"].append(days_waiting)
                    limit_stats["filled"] += 1
                    limit_stats["avg_wait_days"].append(days_waiting)
                    
                    entry_price = limit_data["limit_price"]
                    signal = limit_data["signal"]
                    
                    shares = account.calculate_position_size(entry_price, signal["levels"]["sl"])
                    if shares > 0:
                        open_position = Position(signal, current_date, entry_price, shares)
                        print(f"  ✅ LIMIT FILLED: {limit_data['action']} @ ${entry_price:.2f} (waited {days_waiting} days)")
                    
                    pending_limit = None
                
                elif days_waiting >= LIMIT_ORDER_WAIT_DAYS:
                    ticker_limit_stats["expired"] += 1
                    limit_stats["expired"] += 1
                    print(f"  ⏰ LIMIT EXPIRED: {limit_data['action']} @ ${limit_data['limit_price']:.2f}")
                    pending_limit = None
            
            if not open_position and not pending_limit and idx % 7 == 0 and idx > 0:
                ticker_signal_stats["checks"] += 1
                signal_stats["total_checks"] += 1
                
                try:
                    octopus_signal = analyze_asset_octopus_3agents(ticker, "Краткосрочный", debug=True)
                    action = octopus_signal["recommendation"]["action"]
                    confidence = octopus_signal["recommendation"]["confidence"]
                    entry_kind = octopus_signal.get("entry_kind", "market")
                    
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
                    
                    if idx + 1 < len(historical_data):
                        ticker_validation["total"] += 1
                        validation_stats["total"] += 1
                        
                        historical_subset = historical_data.iloc[:idx+1]
                        atr = calculate_historical_atr(historical_subset, n=14)
                        
                        expected_atr_min = historical_subset['c'].iloc[-1] * 0.005
                        if atr <= expected_atr_min:
                            ticker_validation["fallback_atr"] += 1
                            validation_stats["fallback_atr"] += 1
                        
                        limit_price = octopus_signal["levels"]["entry"]
                        new_levels = recalculate_levels(action, limit_price, atr)
                        
                        if not validate_levels(action, new_levels["entry"], new_levels["sl"], new_levels["tp3"]):
                            ticker_validation["invalid"] += 1
                            validation_stats["invalid"] += 1
                            continue
                        
                        octopus_signal["levels"] = new_levels
                        
                        if entry_kind == "limit":
                            ticker_limit_stats["total"] += 1
                            limit_stats["total_limits"] += 1
                            
                            pending_limit = {
                                "signal": octopus_signal,
                                "action": action,
                                "limit_price": limit_price,
                                "order_date": current_date,
                                "atr": atr
                            }
                        else:
                            next_row = historical_data.iloc[idx + 1]
                            entry_price = next_row["o"]
                            
                            new_levels = recalculate_levels(action, entry_price, atr)
                            octopus_signal["levels"] = new_levels
                            
                            sl_price = new_levels["sl"]
                            
                            if entry_price > 0 and sl_price > 0:
                                shares = account.calculate_position_size(entry_price, sl_price)
                                if shares > 0:
                                    open_position = Position(octopus_signal, current_date, entry_price, shares)
                
                except Exception as e:
                    error_msg = str(e)[:100]
                    print(f"  ⚠️  Error at {current_date}: {error_msg}")
                    continue
        
        if pending_limit:
            ticker_limit_stats["expired"] += 1
            limit_stats["expired"] += 1
        
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
            
            avg_wait = np.mean(ticker_limit_stats["wait_days"]) if ticker_limit_stats["wait_days"] else 0
            fill_rate = (ticker_limit_stats["filled"] / ticker_limit_stats["total"] * 100) if ticker_limit_stats["total"] > 0 else 0
            
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
                "limit_stats": {
                    "total_limits": ticker_limit_stats["total"],
                    "filled": ticker_limit_stats["filled"],
                    "expired": ticker_limit_stats["expired"],
                    "fill_rate_pct": round(fill_rate, 2),
                    "avg_wait_days": round(avg_wait, 2)
                },
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
            
            print(f"\n  📝 Sample trades:")
            for i, trade in enumerate(trades[:3], 1):
                print(f"    {i}. {trade['entry_date']} → {trade['exit_date']}")
                print(f"       ${trade['entry_price']} → ${trade['exit_price']} | {trade['action']}")
                print(f"       Hit: {trade['hit_level']} | P/L: ${trade['profit_dollars']:,.2f} ({trade['profit_pct']*100:.2f}%)")
        else:
            print(f"\n  ❌ Нет сделок")
            print(f"     Сигналов: BUY={ticker_signal_stats['buy']}, SHORT={ticker_signal_stats['short']}, WAIT={ticker_signal_stats['wait']}")
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"octopus_backtest_{timestamp}.json"
    
    avg_wait_days = np.mean(limit_stats["avg_wait_days"]) if limit_stats["avg_wait_days"] else 0
    fill_rate = (limit_stats["filled"] / limit_stats["total_limits"] * 100) if limit_stats["total_limits"] > 0 else 0
    
    with open(output_file, "w") as f:
        json.dump({
            "backtest_date": datetime.now().isoformat(),
            "version": "12.2_alphapulse_fix",
            "description": "M7/W7/AlphaPulse with majority voting, AlphaPulse fix applied",
            "agents": ["M7", "W7", "AlphaPulse"],
            "voting_method": "majority" if USE_MAJORITY_VOTING else "weighted",
            "fix_applied": "Removed use_ml parameter from AlphaPulse call",
            "initial_capital": INITIAL_CAPITAL,
            "tickers": TICKERS,
            "signal_statistics": signal_stats,
            "limit_order_statistics": {
                "total_limits": limit_stats["total_limits"],
                "filled": limit_stats["filled"],
                "expired": limit_stats["expired"],
                "fill_rate_pct": round(fill_rate, 2),
                "avg_wait_days": round(avg_wait_days, 2)
            },
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


if __name__ == "__main__":
    run_octopus_backtest()
