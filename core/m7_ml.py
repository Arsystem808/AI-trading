import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class M7MLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "models/m7_model.pkl"
        self.scaler_path = "models/m7_scaler.pkl"
        self.load_model()

    def load_model(self):
        """Загрузка обученной модели и скейлера"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
        except:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def save_model(self):
        """Сохранение модели и скейлера"""
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def prepare_features(self, df, ticker):
        """Подготовка признаков для ML модели"""
        # Базовые технические индикаторы
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()
        df["momentum"] = df["close"] / df["close"].shift(5) - 1
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["rsi"] = self.calculate_rsi(df["close"])

        # Отношения цен к уровням
        pivots = self.calculate_pivot_levels(df)
        for level_name, level_value in pivots.items():
            df[f"pct_to_{level_name}"] = (df["close"] - level_value) / level_value

        # Объемы
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Удаляем пропущенные значения
        df = df.dropna()

        # Выбираем последние 100 дней для обучения
        recent_data = df.tail(100).copy()

        # Целевая переменная: будет ли цена через 5 дней выше текущей
        recent_data["target"] = (recent_data["close"].shift(-5) > recent_data["close"]).astype(int)

        # Признаки
        feature_columns = ["returns", "volatility", "momentum", "sma_20", "sma_50", "rsi", "volume_ratio"]

        # Добавляем признаки уровней
        for level_name in pivots.keys():
            feature_columns.append(f"pct_to_{level_name}")

        features = recent_data[feature_columns]
        target = recent_data["target"]

        return features, target, feature_columns

    def calculate_rsi(self, series, period=14):
        """Расчет RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_pivot_levels(self, df):
        """Расчет уровней Pivot"""
        # Используем последнюю завершенную неделю для расчета
        weekly = df.resample("W").agg({"high": "max", "low": "min", "close": "last"})
        if len(weekly) < 2:
            return {}

        last_week = weekly.iloc[-2]
        H, L, C = last_week["high"], last_week["low"], last_week["close"]

        P = (H + L + C) / 3
        R1 = (2 * P) - L
        S1 = (2 * P) - H
        R2 = P + (H - L)
        S2 = P - (H - L)

        return {"P": P, "R1": R1, "R2": R2, "S1": S1, "S2": S2}

    def train_model(self, df, ticker):
        """Обучение модели на исторических данных"""
        features, target, feature_columns = self.prepare_features(df, ticker)

        if len(features) < 30:  # Минимальное количество样本
            return False

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Масштабирование признаков
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучение модели
        self.model.fit(X_train_scaled, y_train)

        # Сохранение модели
        self.save_model()

        # Оценка точности
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        print(f"Model trained. Train score: {train_score:.3f}, Test score: {test_score:.3f}")
        return True

    def predict_signal(self, df, ticker):
        """Предсказание сигнала с помощью ML модели"""
        if self.model is None:
            return None

        # Подготовка признаков для последней доступной точки
        features, _, feature_columns = self.prepare_features(df, ticker)
        if len(features) == 0:
            return None

        # Берем последний доступный набор признаков
        latest_features = features.iloc[-1:].copy()

        # Масштабирование
        scaled_features = self.scaler.transform(latest_features)

        # Предсказание
        prediction = self.model.predict_proba(scaled_features)[0]
        p_long = prediction[1]  # Вероятность роста

        return {
            "p_long": float(p_long),
            "confidence": float(min(0.95, max(0.5, p_long * 0.8 + 0.1))),  # Нормализуем уверенность
            "features": latest_features.to_dict("records")[0],
        }
