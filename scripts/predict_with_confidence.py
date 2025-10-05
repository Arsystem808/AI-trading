import json
import sys

import numpy as np

from core.confidence import intervals_from_quantiles, shap_breakdown
from core.model_loader import load_model_for


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_with_confidence.py <TICKER>")
        sys.exit(2)
    ticker = sys.argv[1]
    model = load_model_for(ticker)
    n = getattr(model, "n_features_in_", None)
    x = np.zeros((1, n)) if n else None

    point = None
    if x is not None:
        try:
            point = float(np.ravel(model.predict(x))[0])
        except Exception:
            point = None

    family = "global"
    if isinstance(getattr(model, "objective_", None), str) or hasattr(
        model, "booster_"
    ):
        family = "alphapulse"  # при необходимости адаптировать

    ints = intervals_from_quantiles(ticker, family, x) if x is not None else None
    shap_top = shap_breakdown(model, x)

    out = {
        "ticker": ticker,
        "point": point,
        "interval": ints,
        "confidence": (
            None
            if not ints or point is None
            else max(0.0, 1.0 - (ints["width"] / (abs(point) + 1e-6)))
        ),
        "shap_top": shap_top,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
