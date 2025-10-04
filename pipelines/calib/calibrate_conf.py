import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


DEFAULT_CAL: Dict[str, Any] = {
    "Global": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
    "M7": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
    "W7": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
    "AlphaPulse": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
    "Octopus": {"conf": {"method": "sigmoid", "params": {"a": 1.2, "b": -0.10}}},
}


def load_base(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return DEFAULT_CAL.copy()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_CAL.copy()


def save_base(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Простейшая калибровка b для M7/Octopus")
    ap.add_argument("--data", type=str, required=True, help="Путь к parquet с колонками slope, vol, y")
    ap.add_argument("--out", type=str, default="config/calibration.json", help="Куда сохранить JSON с калибровкой")
    args = ap.parse_args()

    out_path = Path(args.out)
    base = load_base(out_path)

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["atr14", "vol", "slope", "y"])

    # Суррогат «сырая» уверенность
    raw = 0.5 + 0.2 * np.tanh(df["slope"].values * 500.0) - 0.1 * np.clip(df["vol"].values - 0.3, 0, 1)
    raw = np.clip(raw, 0.01, 0.99)
    y = df["y"].values.astype(int)

    # Подбор b по минимуму Brier
    bs_best, b_best = 1e9, 0.0
    for b in np.linspace(-0.3, 0.3, 25):
        p = 1.0 / (1.0 + np.exp(-(1.0 * raw + b)))
        bs = brier_score_loss(y, p)
        if bs < bs_best:
            bs_best, b_best = bs, float(b)

    base.setdefault("M7", {}).setdefault("conf", {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}})
    base["M7"]["conf"]["params"]["b"] = float(b_best)

    save_base(base, out_path)


if __name__ == "__main__":
    main()
