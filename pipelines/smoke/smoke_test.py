import argparse
import json

from core.strategy import analyze_asset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, default="AAPL,MSFT,TSLA")
    ap.add_argument("--strategies", type=str, default="Global,M7,W7,AlphaPulse,Octopus")
    args = ap.parse_args()

    for t in [x.strip().upper() for x in args.tickers.split(",") if x.strip()]:
        for s in [y.strip() for y in args.strategies.split(",") if y.strip()]:
            out = analyze_asset(t, "Краткосрочный", s)
            assert "recommendation" in out and "levels" in out and "probs" in out, "Missing keys"
            assert "action" in out["recommendation"] and "confidence" in out["recommendation"], "Bad reco"
            print(s, t, out["recommendation"], flush=True)


if __name__ == "__main__":
    main()
