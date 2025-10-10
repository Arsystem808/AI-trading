# features/registry.py
from typing import Dict, Any

class FeatureRegistry:
    FEATURES: Dict[str, Dict[str, Any]] = {
        "pivot_fibonacci_v1": {
            "compute_fn": "core.features.technical.calc_pivot_features",
            "dependencies": ["high", "low", "close"],
            "update_freq": "daily",  # для intraday; для daily можно хранить "weekly"
            "params": {"method": "fibonacci", "period_rule": "intraday->D, daily->W"},
            "importance_score": 0.89,
        },
        "macd_regime_v2": {
            "compute_fn": "core.features.technical.calc_macd_regime",
            "dependencies": ["close"],
            "update_freq": "15min",
            "importance_score": 0.76,
        },
    }
