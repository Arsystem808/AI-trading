#!/usr/bin/env python3
# scripts/train_m7pro.py
import argparse

from core.polygon_client import PolygonClient
from core.strategy import M7MLModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--days", type=int, default=720)
    p.add_argument("--n_estimators", type=int, default=600)
    args = p.parse_args()

    df = PolygonClient().daily_ohlc(args.ticker, days=args.days)
    m = M7MLModel()
    info = m.train_and_save(
        df, n_estimators=args.n_estimators, max_depth=None, use_calibration=True
    )
    print("trained:", info)


if __name__ == "__main__":
    main()
