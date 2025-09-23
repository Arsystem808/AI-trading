import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_m7pro(ticker: str, data_path: str):
    """
    Обучение модели M7pro для тикера на основе исторических данных.
    data_path - путь к CSV с признаками и таргетом.
    """

    df = pd.read_csv(data_path)

    # Используйте ваши фичи с префиксом "feature_"
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    target_col = "target"

    if target_col not in df.columns or not feature_cols:
        raise ValueError("Нет необходимых признаков или целевой метки в данных")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(n_estimators=1000, learning_rate=0.05)

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=50,
              verbose=20)

    os.makedirs("models", exist_ok=True)

    model_path = f"models/arxora_m7pro_{ticker}.joblib"
    joblib.dump(model, model_path)
    print(f"Модель сохранена: {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train M7pro model")
    parser.add_argument("--ticker", type=str, required=True, help="Тикер для обучения модели")
    parser.add_argument("--data", type=str, required=True, help="Путь к CSV с обучающими данными")
    args = parser.parse_args()

    train_m7pro(args.ticker, args.data)
