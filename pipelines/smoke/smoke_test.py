import argparse
import json
from typing import Any, Dict

from core.strategy import analyze_asset


def _validate_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("analyze_asset must return dict")
    if not {"recommendation", "levels", "probs"} <= set(payload.keys()):
        raise KeyError("Missing keys in analyze_asset payload")
    reco = payload["recommendation"]
    if not isinstance(reco, dict) or not {"action", "confidence"} <= set(reco.keys()):
        raise KeyError("Bad recommendation payload")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run strategies over tickers")
    ap.add_argument("--tickers", type=str, default="AAPL,MSFT,TSLA")
    ap.add_argument(
        "--strategies",
        type=str,
        default="Global,M7,W7,AlphaPulse,Octopus",
    )
    args = ap.parse_args()

    tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    strategies = [y.strip() for y in args.strategies.split(",") if y.strip()]

    for t in tickers:
        for s in strategies:
            out = analyze_asset(t, "Краткосрочный", s)
            _validate_payload(out)
            print(
                json.dumps(
                    {"strategy": s, "ticker": t, "recommendation": out["recommendation"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
