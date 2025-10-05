from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PivotLevels:
    P: float
    R1: float
    R2: float
    R3: float
    S1: float
    S2: float
    S3: float


def fib_pivots(prev_high: float, prev_low: float, prev_close: float) -> PivotLevels:
    P = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    R1 = P + 0.382 * rng
    R2 = P + 0.618 * rng
    R3 = P + 1.000 * rng
    S1 = P - 0.382 * rng
    S2 = P - 0.618 * rng
    S3 = P - 1.000 * rng
    return PivotLevels(P, R1, R2, R3, S1, S2, S3)


def last_consecutive_run(series_bool: pd.Series) -> int:
    if series_bool.empty:
        return 0
    last_val = series_bool.iloc[-1]
    cnt = 0
    for v in reversed(series_bool.tolist()):
        if v == last_val:
            cnt += 1
        else:
            break
    return cnt


def pct_diff(a: float, b: float) -> float:
    if b == 0:
        return np.inf
    return abs(a - b) / abs(b)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))
