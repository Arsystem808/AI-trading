import pandas as pd
from core.polygon_client import PolygonClient
from core.m7_ml import M7MLModel

def train_models_for_tickers(tickers):
    """Обучение моделей для списка тикеров"""
    cli = PolygonClient()
    ml_model = M7MLModel()
    
    for ticker in tickers:
        print(f"Training model for {ticker}...")
        try:
            # Получаем исторические данные
            df = cli.daily_ohlc(ticker, days=365)  # Год данных для обучения
            
            # Обучаем модель
            success = ml_model.train_model(df, ticker)
            
            if success:
                print(f"Model for {ticker} trained successfully")
            else:
                print(f"Not enough data for {ticker}")
                
        except Exception as e:
            print(f"Error training model for {ticker}: {e}")

if __name__ == "__main__":
    # Список тикеров для обучения
    popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]
    train_models_for_tickers(popular_tickers)
