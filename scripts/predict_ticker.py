import sys

import numpy as np

from core.model_loader import load_model_for


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_ticker.py <TICKER>")
        sys.exit(2)
    ticker = sys.argv[1]
    model = load_model_for(ticker)
    n = getattr(model, "n_features_in_", None)
    if n is None:
        print(
            f"Loaded {type(model).__name__} for {ticker}, skip dry-run (no n_features_in_)"
        )
        return
    x = np.zeros((1, n))
    pred = model.predict(x)
    print(
        f"OK: {ticker} -> {type(model).__name__}, n_features_in_={n}, pred_shape={getattr(pred, 'shape', None)}"
    )


if __name__ == "__main__":
    main()
