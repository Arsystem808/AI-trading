import os
import pandas as pd
import requests
import ta
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import joblib

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


def download_polygon_data(ticker, from_date, to_date):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{from_date}/{to_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["results"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(
        columns={"c": "close", "o": "open", "h": "high", "l": "low", "v": "volume"},
        inplace=True,
    )
    return df[["date", "open", "high", "low", "close", "volume"]]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_10"] = ta.trend.sma_indicator(df["close"], window=10)
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    return df.dropna()


def prepare_training_data(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    df = download_polygon_data(ticker, from_date, to_date)
    df = add_technical_features(df)
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    df = df.dropna()
    return df


def train_and_save_model(ticker: str, df: pd.DataFrame) -> str:
    feature_cols = ["sma_10", "sma_20", "rsi"]
    target_col = "target"

    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(n_estimators=1000, learning_rate=0.05)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"arxora_m7_{ticker}.joblib")
    joblib.dump(model, model_path)
    print(f"Модель сохранена: {model_path}")
    return model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Скачать данные, подготовить и обучить модель M7 на Polygon"
    )
    parser.add_argument("--ticker", type=str, required=True, help="Тикер, например SPY")
    parser.add_argument(
        "--from_date", type=str, required=True, help="Дата начала YYYY-MM-DD"
    )
    parser.add_argument(
        "--to_date", type=str, required=True, help="Дата конца YYYY-MM-DD"
    )
    args = parser.parse_args()

    data = prepare_training_data(args.ticker, args.from_date, args.to_date)
    train_and_save_model(args.ticker, data)
