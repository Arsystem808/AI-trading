# etl/transform.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("etl.transform")


class PivotCalculator:
    """Расчёт Pivot уровней для разных таймфреймов (Fibonacci по умолчанию)"""

    @staticmethod
    def _infer_period(index: pd.DatetimeIndex) -> str:
        """Определяет период агрегации предыдущих H/L/C для Pivot."""
        freq = pd.infer_freq(index)
        if freq is None:
            # эвристика: если шаг < 1 дня — считаем intraday
            is_intraday = (index[1] - index[0]) < pd.Timedelta(days=1)
            return "D" if is_intraday else "W"
        intraday_aliases = {"T", "min", "H"}
        if any(a in freq.upper() for a in ["T", "MIN", "H"]):
            return "D"
        return "W"

    @staticmethod
    def calculate_levels(df: pd.DataFrame, method: str = "fibonacci") -> pd.DataFrame:
        """
        Рассчитывает Pivot уровни для каждой свечи на основе предыдущего периода.

        Args:
            df: DataFrame с OHLC данными (DatetimeIndex, open/high/low/close/volume).
            method: 'fibonacci' (по умолчанию), 'classic' — классика в качестве fallback.
        """
        df = df.copy()

        # Проверка входа
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLC columns: {missing}")

        # Определяем период агрегации
        period = PivotCalculator._infer_period(df.index)

        # Группируем по периоду, берём пред. период H/L/C
        grouped = df.groupby(pd.Grouper(freq=period))
        prev_high = grouped["high"].max().shift(1)
        prev_low = grouped["low"].min().shift(1)
        prev_close = grouped["close"].last().shift(1)

        # Протягиваем на исходный таймфрейм
        prev_high = prev_high.reindex(df.index, method="ffill")
        prev_low = prev_low.reindex(df.index, method="ffill")
        prev_close = prev_close.reindex(df.index, method="ffill")

        # Центральный Pivot
        pivot = (prev_high + prev_low + prev_close) / 3.0
        df["pivot"] = pivot
        rng = (prev_high - prev_low)

        if method.lower() == "fibonacci":
            # Fibonacci Pivot Points
            df["r1"] = pivot + 0.382 * rng
            df["r2"] = pivot + 0.618 * rng
            df["r3"] = pivot + 1.000 * rng
            df["s1"] = pivot - 0.382 * rng
            df["s2"] = pivot - 0.618 * rng
            df["s3"] = pivot - 1.000 * rng
        else:
            # Классические уровни (fallback)
            df["r1"] = 2 * pivot - prev_low
            df["s1"] = 2 * pivot - prev_high
            df["r2"] = pivot + (prev_high - prev_low)
            df["s2"] = pivot - (prev_high - prev_low)
            df["r3"] = prev_high + 2 * (pivot - prev_low)
            df["s3"] = prev_low - 2 * (prev_high - pivot)

        # Расстояния до уровней (в % от цены)
        for level in ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"]:
            df[f"dist_to_{level}"] = (df["close"] - df[level]) / df["close"] * 100.0
            df[f"dist_to_{level}_abs"] = df[f"dist_to_{level}"].abs()

        # Ближайший уровень среди ключевых
        df["nearest_level"] = (
            df[["dist_to_pivot_abs", "dist_to_r1_abs", "dist_to_s1_abs"]]
            .idxmin(axis=1)
            .str.replace("dist_to_", "", regex=False)
            .str.replace("_abs", "", regex=False)
        )

        return df


class MACDAnalyzer:
    """Продвинутый MACD анализ для ETL и ML"""

    @staticmethod
    def calculate(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """Расчёт MACD + расширенные ML-фичи."""
        df = df.copy()

        ema_fast = df["close"].ewm(span=fast, min_periods=fast).mean()
        ema_slow = df["close"].ewm(span=slow, min_periods=slow).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, min_periods=signal).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        df["macd_cross"] = np.where(df["macd"] > df["macd_signal"], 1, -1)
        df["macd_cross_change"] = df["macd_cross"].diff()

        # Длительность "окраски" гистограммы (серии плюсов/минусов)
        df["macd_color_length"] = (
            df.groupby((df["macd_cross"] != df["macd_cross"].shift()).cumsum())
            .cumcount()
            + 1
        )

        # Ускорения
        df["macd_acceleration"] = df["macd"].diff()
        df["macd_hist_acceleration"] = df["macd_hist"].diff()

        # Нормализации
        df["macd_normalized"] = df["macd"] / df["close"]
        df["macd_hist_normalized"] = df["macd_hist"] / df["close"]

        # Роллинг статистика
        for window in [5, 10, 20]:
            df[f"macd_hist_mean_{window}"] = df["macd_hist"].rolling(window).mean()
            df[f"macd_hist_std_{window}"] = df["macd_hist"].rolling(window).std()

        # Z-score экстремумов
        roll_mean = df["macd_hist"].rolling(20).mean()
        roll_std = df["macd_hist"].rolling(20).std()
        df["macd_hist_zscore"] = (df["macd_hist"] - roll_mean) / roll_std

        return df


class MarketRegimeDetector:
    """Определение рыночного режима для адаптивной торговли"""

    @staticmethod
    def detect(df: pd.DataFrame) -> pd.DataFrame:
        """Определяет текущий режим рынка: uptrend/downtrend/ranging/volatile/neutral."""
        df = df.copy()

        # ATR proxy
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(14).mean()
        df["atr_normalized"] = df["atr"] / df["close"]

        # ADX
        df["adx"] = MarketRegimeDetector._calculate_adx(df)

        # Режимы
        conditions = [
            (df["adx"] > 25) & (df["macd_hist"] > 0),
            (df["adx"] > 25) & (df["macd_hist"] < 0),
            (df["adx"] < 20) & (df["atr_normalized"] < df["atr_normalized"].quantile(0.3)),
            (df["atr_normalized"] > df["atr_normalized"].quantile(0.7)),
        ]
        choices = ["uptrend", "downtrend", "ranging", "volatile"]
        df["market_regime"] = np.select(conditions, choices, default="neutral")

        # One-hot для ML
        regime_dummies = pd.get_dummies(df["market_regime"], prefix="regime")
        df = pd.concat([df, regime_dummies], axis=1)

        return df

    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX по стандартной схеме."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat(
            [high - low, np.abs(high - close.shift()), np.abs(low - close.shift())],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx


class SignalGenerator:
    """Генерация торговых сигналов на основе Pivot + MACD + эвристик ML"""

    def __init__(self, conservative: bool = False):
        self.conservative = conservative

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает df с сигналами, таргетами и confidence."""
        df = df.copy()

        # Правило Pivot + MACD
        df["signal_pivot_macd"] = self._pivot_macd_signals(df)

        # Простая ML-эвристика (можно заменить моделью)
        df["signal_ml_score"] = self._ml_based_signals(df)

        # Комбинация
        df["signal_combined"] = self._combine_signals(df)

        # Будущие доходности как таргеты
        df["future_return_1"] = df["close"].shift(-1) / df["close"] - 1
        df["future_return_5"] = df["close"].shift(-5) / df["close"] - 1
        df["future_return_10"] = df["close"].shift(-10) / df["close"] - 1

        # Классификация таргета
        threshold = 0.005 if self.conservative else 0.002
        df["target_1"] = np.where(
            df["future_return_1"] > threshold,
            1,
            np.where(df["future_return_1"] < -threshold, -1, 0),
        )
        df["target_5"] = np.where(
            df["future_return_5"] > threshold * 3,
            1,
            np.where(df["future_return_5"] < -threshold * 3, -1, 0),
        )

        # Уверенность
        df["signal_confidence"] = self._calculate_confidence(df)

        return df

    def _pivot_macd_signals(self, df: pd.DataFrame) -> pd.Series:
        """Правило: у поддержки + MACD растёт — buy, у сопротивления + MACD падает — sell."""
        signals = pd.Series(index=df.index, data=0, dtype="int32")

        buy_condition = (
            (df.get("dist_to_s1_abs", pd.Series(index=df.index, data=np.nan)) < 0.25)
            & (df["macd_hist"] > 0)
            & (df["macd_cross_change"] == 2)
        )

        sell_condition = (
            (df.get("dist_to_r1_abs", pd.Series(index=df.index, data=np.nan)) < 0.25)
            & (df["macd_hist"] < 0)
            & (df["macd_cross_change"] == -2)
        )

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

    def _ml_based_signals(self, df: pd.DataFrame) -> pd.Series:
        """Простая score-эвристика — placeholder для модели."""
        score = pd.Series(index=df.index, data=0.0)

        score += df["macd_hist_zscore"].clip(-2, 2) * 0.2
        score += np.where(
            df["market_regime"] == "uptrend",
            0.3,
            np.where(df["market_regime"] == "downtrend", -0.3, 0.0),
        )
        score -= df.get("dist_to_pivot_abs", 0.0) * 0.01

        return score

    def _combine_signals(self, df: pd.DataFrame) -> pd.Series:
        """Взвешенная комбинация правил и score."""
        combined = df["signal_pivot_macd"] * 0.6 + df["signal_ml_score"] * 0.4
        return np.where(combined > 0.3, 1, np.where(combined < -0.3, -1, 0))

    def _calculate_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Оценка уверенности сигнала."""
        confidence = pd.Series(index=df.index, data=0.5, dtype="float64")
        confidence += np.abs(df["macd_hist_zscore"].clip(-2, 2)) * 0.1
        confidence += np.where(df["macd_color_length"] > 5, 0.1, 0)
        confidence += np.where(df["adx"] > 25, 0.2, 0)
        return confidence.clip(0, 1)


class AdvancedTransformer:
    """
    Главный ETL Transformer для ML Pipeline:
    - Валидация входа/выхода
    - Базовые фичи, Fibonacci Pivots, MACD, рыночный режим
    - Сигналы/таргеты, лаги, фичи для training/inference
    """

    def __init__(self, mode: str = "training", feature_version: str = "v1"):
        if mode not in {"training", "inference"}:
            raise ValueError("mode must be 'training' or 'inference'")
        self.mode = mode
        self.feature_version = feature_version
        self.pivot_calc = PivotCalculator()
        self.macd_analyzer = MACDAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.signal_gen = SignalGenerator()
        logger.info(f"Transformer initialized in {mode} mode, features {feature_version}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Полная трансформация входных OHLC-данных в ML-ready датасет."""
        logger.info(f"Starting transformation for {len(df)} rows")

        self._validate_input(df)

        # Базовые фичи
        df = self._add_basic_features(df)

        # Технические индикаторы
        df = self.pivot_calc.calculate_levels(df, method="fibonacci")
        df = self.macd_analyzer.calculate(df)
        df = self.regime_detector.detect(df)

        # Сигналы и таргеты
        df = self.signal_gen.generate_signals(df)

        # Временные фичи
        df = self._add_time_features(df)

        # Лаговые фичи
        df = self._add_lag_features(df)

        # Режим-специфика
        if self.mode == "training":
            df = self._add_training_features(df)
        else:
            df = self._add_inference_features(df)

        # Финальная валидация/очистка
        df = self._validate_output(df)

        logger.info(f"Transformation complete. Final shape: {df.shape}")
        return df

    def _validate_input(self, df: pd.DataFrame):
        required_columns = ["open", "high", "low", "close", "volume"]
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(df) < 50:
            raise ValueError(f"Need at least 50 rows, got {len(df)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["dollar_volume"] = df["close"] * df["volume"]
        df["hl_ratio"] = df["high"] / df["low"]
        df["close_to_high"] = df["close"] / df["high"]
        df["close_to_low"] = df["close"] / df["low"]
        df["volatility_20"] = df["returns"].rolling(20).std()
        df["volatility_5"] = df["returns"].rolling(5).std()
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["week_of_year"] = df.index.isocalendar().week.astype(int)
        df["is_market_open"] = df["hour"].between(9, 16)
        df["is_pre_market"] = df["hour"].between(4, 9)
        df["is_after_hours"] = df["hour"].between(16, 20)
        return df

    def _add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        df = df.copy()
        important_features = ["returns", "macd_hist", "signal_combined", "volatility_5", "dollar_volume"]
        for feature in important_features:
            if feature in df.columns:
                for lag in lags:
                    df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
        return df

    def _add_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        window = 20
        df["correlation_macd_price"] = df["macd"].rolling(window).corr(df["close"].pct_change())
        return df

    def _add_inference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # значения последнего сигнала/уверенности для быстрой телеметрии
        df["last_signal"] = df["signal_combined"].iloc[-1]
        df["last_confidence"] = df["signal_confidence"].iloc[-1]
        return df

    def _validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method="bfill", limit=50)
        if self.mode == "training":
            df = df.dropna()
        else:
            df = df.fillna(method="ffill")
        return df

    def get_feature_names(self) -> list:
        exclude = [
            "open", "high", "low", "close", "volume",
            "target_1", "target_5", "target_10",
            "future_return_1", "future_return_5", "future_return_10",
        ]
        # синтетический сэмпл для вычисления списка признаков
        sample_df = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=100, freq="15min"),
            data={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1_000_000},
        )
        transformed = AdvancedTransformer(mode="training").transform(sample_df)
        features = [col for col in transformed.columns if col not in exclude]
        return features

    def get_last_signal(self, df: pd.DataFrame) -> Dict:
        df = self.transform(df)
        last_row = df.iloc[-1]
        return {
            "timestamp": df.index[-1],
            "signal": int(last_row["signal_combined"]),
            "confidence": float(last_row["signal_confidence"]),
            "close": float(last_row["close"]),
            "pivot": float(last_row["pivot"]),
            "r1": float(last_row["r1"]),
            "s1": float(last_row["s1"]),
            "macd_hist": float(last_row["macd_hist"]),
            "market_regime": last_row["market_regime"],
            "comment": self._generate_comment(last_row),
        }

    def _generate_comment(self, row: pd.Series) -> str:
        signal_map = {1: "BUY", -1: "SHORT", 0: "WAIT"}
        signal = signal_map[int(row["signal_combined"])]

        comments = []

        # Pivot контекст
        if row.get("dist_to_pivot_abs", 1e6) < 0.5:
            comments.append(f"Цена у Pivot ({row['pivot']:.2f})")
        elif row.get("nearest_level") in ["r1", "r2", "r3"]:
            comments.append(f"Цена у сопротивления {row['nearest_level'].upper()}")
        elif row.get("nearest_level") in ["s1", "s2", "s3"]:
            comments.append(f"Цена у поддержки {row['nearest_level'].upper()}")

        # MACD контекст
        if abs(row.get("macd_hist_zscore", 0)) > 2:
            comments.append(f"MACD экстремальный ({row['macd_hist_zscore']:.1f} σ)")
        elif row.get("macd_color_length", 0) > 10:
            comments.append(f"MACD импульс длится {int(row['macd_color_length'])} свечей")

        # Режим рынка
        comments.append(f"Режим: {row.get('market_regime', 'n/a')}")

        # Уверенность
        conf = row.get("signal_confidence", 0.0)
        level = "высокая" if conf > 0.7 else "средняя" if conf > 0.4 else "низкая"
        comments.append(f"Уверенность: {level} ({conf:.0%})")

        return f"{signal} | {' | '.join(comments)}"


# ==========================
#     USAGE EXAMPLES
# ==========================
if __name__ == "__main__":
    # Пример использования (демо на синтетике)
    dates = pd.date_range("2024-01-01", periods=1000, freq="15min")
    test_df = pd.DataFrame(
        index=dates,
        data={
            "open": np.random.randn(1000).cumsum() + 100,
            "high": np.random.randn(1000).cumsum() + 101,
            "low": np.random.randn(1000).cumsum() + 99,
            "close": np.random.randn(1000).cumsum() + 100,
            "volume": np.random.randint(100_000, 1_000_000, 1000),
        },
    )

    # Training mode
    transformer = AdvancedTransformer(mode="training")
    train_df = transformer.transform(test_df)
    print(f"[TRAIN] Shape: {train_df.shape}")

    features = transformer.get_feature_names()
    X = train_df[features]
    y = train_df["target_5"]
    print(f"[TRAIN] X: {X.shape}, y counts: {y.value_counts().to_dict()}")

    # Inference mode
    transformer_live = AdvancedTransformer(mode="inference")
    signal = transformer_live.get_last_signal(test_df)
    print(f"[LIVE] Signal: {signal['signal']}, Confidence: {signal['confidence']:.1%}")
    print(f"[LIVE] Comment: {signal['comment']}")
