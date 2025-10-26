#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Octopus Backtest v12.3 — MARKET ORDERS
Supports two modes:
  - octopus: ensemble of M7/W7/AlphaPulse with majority voting
  - individual: single agent backtest (M7 or W7 or AlphaPulse)
Produces JSON report to --output if provided; otherwise to results/<prefix>_backtest_<ts>.json
"""

import argparse
import json
import os
from datetime import datetime, date
from pathlib import Path

# ----------------------
# Imports from your codebase
# ----------------------
# These should already exist in the repo
from core.strategy import PolygonClient, CAL_CONF  # noqa: F401
from core.strategy import backtest_octopus, backtest_single_agent  # implement if not yet
from agents.m7 import M7  # noqa: F401
from agents.w7 import W7  # noqa: F401
from agents.alphapulse import AlphaPulse  # noqa: F401

# ----------------------
# Helpers
# ----------------------
def utc_ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def default_output(prefix: str) -> Path:
    Path("results").mkdir(exist_ok=True)
    return Path(f"results/{prefix}_backtest_{utc_ts()}.json")

def load_agent(name: str):
    name = name.upper()
    if name == "M7":
        return M7()
    if name == "W7":
        return W7()
    if name == "ALPHAPULSE":
        return AlphaPulse()
    raise ValueError(f"Unknown agent: {name}")

def to_json_file(obj, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(out_path)

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["octopus", "individual"], default="octopus")
    parser.add_argument("--strategy", choices=["M7", "W7", "AlphaPulse"], default="M7")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--format", choices=["json"], default="json")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    start = parse_date(args.start)
    end = parse_date(args.end)

    meta = {
        "backtest_date": datetime.utcnow().isoformat(),
        "version": "12.3_market_orders",
        "order_type": "market",
        "tickers": tickers,
        "window": {"start": args.start, "end": args.end},
    }

    if args.mode == "individual":
        agent = load_agent(args.strategy)
        results = backtest_single_agent(agent, tickers, start, end, order_type="market")
        report = {
            "strategy": args.strategy,
            "mode": "individual",
            "description": f"{args.strategy} single-agent backtest (market orders)",
            **meta,
            "results": results["per_ticker"],
            "summary": results.get("summary", {}),
        }
        out_path = Path(args.output) if args.output else default_output(args.strategy.lower())
    else:
        # octopus ensemble
        agents = [M7(), W7(), AlphaPulse()]
        results = backtest_octopus(agents, tickers, start, end, voting="majority", order_type="market")
        report = {
            "agents": ["M7", "W7", "AlphaPulse"],
            "mode": "octopus",
            "voting_method": "majority",
            "description": "M7/W7/AlphaPulse ensemble with majority voting (market orders)",
            **meta,
            "results": results["per_ticker"],
            "signal_statistics": results.get("signal_statistics", {}),
            "validation_statistics": results.get("validation_statistics", {}),
            "summary": results.get("summary", {}),
        }
        out_path = Path(args.output) if args.output else default_output("octopus")

    final_path = to_json_file(report, out_path)
    print(f"Results saved to: {final_path}")

if __name__ == "__main__":
    main()
