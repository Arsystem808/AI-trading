# jobs/calibrate.py
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PERF_PATH = os.environ.get("PERF_ALL_PATH", "data/perf/all.csv")
CAL_PATH = os.environ.get("CALIB_PATH", "config/calibration.json")


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    probs = np.clip(probs.astype(float), 0.0, 1.0)
    labels = labels.astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    N = len(labels)
    if N == 0:
        return 0.0
    total = 0.0
    for i in range(bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1])
        if not np.any(m):
            continue
        acc = labels[m].mean()
        conf = probs[m].mean()
        total += (m.sum() / N) * abs(acc - conf)
    return float(total)


def platt_grid(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # Ищем a,b в p' = 1/(1+exp(-(a*p + b))) по минимальному ECE
    best = (1.0, 0.0, 1e9)
    for a in np.linspace(0.6, 1.8, 25):
        for b in np.linspace(-0.4, 0.4, 25):
            z = a * probs + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
            e = ece(p, labels, bins=10)
            if e < best[2]:
                best = (float(a), float(b), float(e))
    return {"a": best[0], "b": best[1], "ece": best[2]}


def read_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_cal(cal: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cal(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {
            "Global": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
            "M7": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
            "W7": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
            "AlphaPulse": {
                "conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}
            },
            "Octopus": {"conf": {"method": "sigmoid", "params": {"a": 1.2, "b": -0.10}}},
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "Global": {"conf": {"method": "sigmoid",
