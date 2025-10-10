from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

import math
import numpy as np
import pandas as pd


__all__ = [
    "PivotLevels",
    "fib_pivots",
    "fib_pivots_from_row",
    "last_consecutive_run",
    "pct_diff",
    "pct_diff_symmetric",
    "clamp",
]


@dataclass(frozen=True, slots=True)
class PivotLevels:
    """
    Набор уровней Fibonacci-пивотов для предыдущего периода:
    - P  — центральный пивот: (high + low + close) / 3
    - R1 — P + 0.382 * (high - low)
    - R2 — P + 0.618 * (high - low)
    - R3 — P + 1.000 * (high - low)
    - S1 — P - 0.382 * (high - low)
    - S2 — P - 0.618 * (high - low)
    - S3 — P - 1.000 * (high - low)

    Формулы соответствуют стандартному определению Fibonacci Pivot Points [1][2].

    References:
    [1] Morpher: Pivot Point Fibonacci (PP, R1/R2/S1/S2 через 0.382/0.618) [web:5616]
    [2] True Fibonacci Pivot Points формулы R1/R2/R3 и S1/S2/S3 [web:5618]
    """
    P: float
    R1: float
    R2: float
    R3: float
    S1: float
    S2: float
    S3: float

    def as_dict(self) -> dict[str, float]:
        """Сериализация уровней в словарь для последующего логгирования/JSON [web:5618]."""
        return {
            "P": self.P,
            "R1": self.R1,
            "R2": self.R2,
            "R3": self.R3,
            "S1": self.S1,
            "S2": self.S2,
            "S3": self.S3,
        }


def _validate_hlc(prev_high: float, prev_low: float, prev_close: float) -> None:
    """Проверка входных значений на корректность и финитность чисел [web:5618]."""
    if not all(map(np.isfinite, (prev_high, prev_low, prev_close))):
        raise ValueError("high/low/close must be finite numbers")
    if prev_high < prev_low:
        raise ValueError("high must be >= low")


def fib_pivots(prev_high: float, prev_low: float, prev_close: float) -> PivotLevels:
    """
    Расчёт Fibonacci Pivot Levels из high/low/close предыдущего периода [web:5616].

    Формулы:
      P  = (H + L + C) / 3
      R1 = P + 0.382 * (H - L)
      R2 = P + 0.618 * (H - L)
      R3 = P + 1.000 * (H - L)
      S1 = P - 0.382 * (H - L)
      S2 = P - 0.618 * (H - L)
      S3 = P - 1.000 * (H - L)
    """
    _validate_hlc(prev_high, prev_low, prev_close)
    P = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    R1 = P + 0.382 * rng
    R2 = P + 0.618 * rng
    R3 = P + 1.000 * rng
    S1 = P - 0.382 * rng
    S2 = P - 0.618 * rng
    S3 = P - 1.000 * rng
    return PivotLevels(P, R1, R2, R3, S1, S2, S3)


def fib_pivots_from_row(
    row: Mapping[str, Any],
    high: str = "high",
    low: str = "low",
    close: str = "close",
) -> PivotLevels:
    """
    Удобный конструктор уровней из строки DataFrame/словаря с ключами high/low/close [web:5618].
    """
    return fib_pivots(float(row[high]), float(row[low]), float(row[close]))


def last_consecutive_run(series_bool: pd.Series) -> int:
    """
    Длина последней непрерывной серии одинаковых значений булевой серии
    (например, сколько подряд True или False в хвосте) [web:5632].

    Реализация: считаем группы смен значений через сдвиг и cumsum, затем
    возвращаем размер последней группы, что масштабируется линейно по N [web:5626].
    """
    if series_bool is None or len(series_bool) == 0:
        return 0
    s = series_bool.astype(bool)
    # помечаем границы серий, накапливаем идентификатор группы
    grp = (s != s.shift(fill_value=~s.iloc[0])).cumsum()
    last_id = grp.iloc[-1]
    return int((grp == last_id).sum())


def pct_diff(a: float, b: float) -> float:
    """
    Относительное отклонение от базового значения b: |a - b| / |b|,
    полезно для «процентного изменения» относительно базы [web:5633].

    Возвращает +inf, если b == 0 или b не финитен, чтобы явно сигнализировать о некорректной базе [web:5633].
    """
    if not np.isfinite(b) or b == 0:
        return np.inf
    return float(abs(a - b) / abs(b))


def pct_diff_symmetric(a: float, b: float) -> float:
    """
    Симметричная «percent difference»: |a - b| / ((|a| + |b|) / 2),
    устойчива к перестановке аргументов и часто используется как
    «процентная разница» в справочниках [web:5624].

    Если оба значения равны нулю, возвращает 0.0 как нулевую разницу [web:5624].
    """
    denom = (abs(a) + abs(b)) / 2.0
    if denom == 0:
        return 0.0
    return float(abs(a - b) / denom)


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Ограничение значения x в пределах [lo, hi] с приведением к float [web:5618].
    """
    if lo > hi:
        lo, hi = hi, lo
    return float(max(lo, min(hi, x)))
