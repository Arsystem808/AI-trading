#!/usr/bin/env python3
"""
Compare backtest results from M7, W7, and AlphaPulse
"""

import json
import pandas as pd
from pathlib import Path

def compare_backtests():
    results_dir = Path("results")
    
    # Find latest results
    m7_file = sorted(results_dir.glob("m7_backtest_*.json"))[-1]
    w7_file = sorted(results_dir.glob("w7_backtest_*.json"))[-1]
    alpha_file = sorted(results_dir.glob("alphapulse_backtest_*.json"))[-1]
    
    # Load data
    m7 = json.load(open(m7_file))
    w7 = json.load(open(w7_file))
    alpha = json.load(open(alpha_file))
    
    # Overall comparison
    print("\n" + "="*80)
    print("📊 STRATEGY COMPARISON")
    print("="*80 + "\n")
    
    comparison = {
        "Strategy": ["M7", "W7", "AlphaPulse"],
        "Total Trades": [
            sum(r["total_trades"] for r in m7["results"]) if m7["results"] else 0,
            sum(r["total_trades"] for r in w7["results"]) if w7["results"] else 0,
            sum(r["total_trades"] for r in alpha["results"]) if alpha["results"] else 0
        ],
        "BUY Signals": [
            m7["signal_statistics"]["buy"],
            w7["signal_statistics"]["buy"],
            alpha["signal_statistics"]["buy"]
        ],
        "SHORT Signals": [
            m7["signal_statistics"]["short"],
            w7["signal_statistics"]["short"],
            alpha["signal_statistics"]["short"]
        ],
        "WAIT Signals": [
            m7["signal_statistics"]["wait"],
            w7["signal_statistics"]["wait"],
            alpha["signal_statistics"]["wait"]
        ]
    }
    
    # Calculate averages
    if m7["results"]:
        comparison["Avg Sharpe"] = [
            round(sum(r["sharpe_ratio"] for r in m7["results"]) / len(m7["results"]), 2),
            round(sum(r["sharpe_ratio"] for r in w7["results"]) / len(w7["results"]), 2) if w7["results"] else 0,
            round(sum(r["sharpe_ratio"] for r in alpha["results"]) / len(alpha["results"]), 2) if alpha["results"] else 0
        ]
        comparison["Avg Win Rate %"] = [
            round(sum(r["win_rate_pct"] for r in m7["results"]) / len(m7["results"]), 1),
            round(sum(r["win_rate_pct"] for r in w7["results"]) / len(w7["results"]), 1) if w7["results"] else 0,
            round(sum(r["win_rate_pct"] for r in alpha["results"]) / len(alpha["results"]), 1) if alpha["results"] else 0
        ]
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    
    # Per-ticker breakdown
    print("\n" + "="*80)
    print("📈 PER-TICKER PERFORMANCE")
    print("="*80 + "\n")
    
    tickers = ["X:BTCUSD", "X:ETHUSD", "AAPL", "NVDA"]
    
    for ticker in tickers:
        m7_ticker = next((r for r in m7["results"] if r["ticker"] == ticker), None)
        w7_ticker = next((r for r in w7["results"] if r["ticker"] == ticker), None)
        alpha_ticker = next((r for r in alpha["results"] if r["ticker"] == ticker), None)
        
        if not any([m7_ticker, w7_ticker, alpha_ticker]):
            continue
        
        print(f"\n{ticker}:")
        ticker_comp = {
            "Strategy": ["M7", "W7", "AlphaPulse"],
            "Trades": [
                m7_ticker["total_trades"] if m7_ticker else 0,
                w7_ticker["total_trades"] if w7_ticker else 0,
                alpha_ticker["total_trades"] if alpha_ticker else 0
            ],
            "Return %": [
                m7_ticker["total_return_pct"] if m7_ticker else 0,
                w7_ticker["total_return_pct"] if w7_ticker else 0,
                alpha_ticker["total_return_pct"] if alpha_ticker else 0
            ],
            "Sharpe": [
                m7_ticker["sharpe_ratio"] if m7_ticker else 0,
                w7_ticker["sharpe_ratio"] if w7_ticker else 0,
                alpha_ticker["sharpe_ratio"] if alpha_ticker else 0
            ],
            "Win Rate %": [
                m7_ticker["win_rate_pct"] if m7_ticker else 0,
                w7_ticker["win_rate_pct"] if w7_ticker else 0,
                alpha_ticker["win_rate_pct"] if alpha_ticker else 0
            ]
        }
        
        df_ticker = pd.DataFrame(ticker_comp)
        print(df_ticker.to_string(index=False))
    
    # Recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80 + "\n")
    
    sharpes = [
        (sum(r["sharpe_ratio"] for r in m7["results"]) / len(m7["results"]) if m7["results"] else -999, "M7"),
        (sum(r["sharpe_ratio"] for r in w7["results"]) / len(w7["results"]) if w7["results"] else -999, "W7"),
        (sum(r["sharpe_ratio"] for r in alpha["results"]) / len(alpha["results"]) if alpha["results"] else -999, "AlphaPulse")
    ]
    
    best_strategy = max(sharpes, key=lambda x: x[0])
    
    if best_strategy[0] > 0.3:
        print(f"✅ Use {best_strategy[1]} ALONE (Sharpe: {best_strategy[0]:.2f})")
        print(f"   This strategy has positive risk-adjusted returns.")
    elif best_strategy[0] > 0:
        print(f"⚠️  {best_strategy[1]} is best but marginal (Sharpe: {best_strategy[0]:.2f})")
        print(f"   Consider improvements before live trading.")
    else:
        print(f"❌ All strategies have negative Sharpe ratios")
        print(f"   Required: Add trend filters, wider stops, or different entry logic")

if __name__ == "__main__":
    compare_backtests()
