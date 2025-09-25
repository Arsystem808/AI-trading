import joblib
import pandas as pd
import numpy as np
from core.polygon_client import PolygonClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

# Основная функция для вычисления технических признаков
def build_feats(df: pd.DataFrame) -> pd.DataFrame:
    # Расчет pivots (пример)
    P = (df['high'] + df['low'] + df['close']) / 3.0
    R1 = 2 * P - df['low']
    S1 = 2 * P - df['high']
    R2 = P + (df['high'] - df['low'])
    S2 = P - (df['high'] - df['low'])

    out = pd.DataFrame(index=df.index)
    out['momentum'] = df['close'].pct_change(5)
    out['returns'] = df['close'].pct_change()
    out['rsi'] = rsi(df['close'])
    out['sma_20'] = df['close'].rolling(20).mean()
    out['sma_50'] = df['close'].rolling(50).mean()
    out['pct_to_P'] = (df['close'] - P) / P
    out['pct_to_R1'] = (R1 - df['close']) / df['close']
    out['pct_to_R2'] = (R2 - df['close']) / df['close']
    out['pct_to_S1'] = (df['close'] - S1) / df['close']
    out['pct_to_S2'] = (df['close'] - S2) / df['close']
    out['volatility'] = out['returns'].rolling(20).std()
    out['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    return out.dropna()


def rsi(series: pd.Series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


class MLModel:
    def __init__(self):
        self.model = joblib.load('models/m7_model.pkl')
        self.scaler = joblib.load('models/m7_scaler.pkl')

    def predict(self, df):
        try:
            X = build_feats(df)
            if X.empty:
                return None

            # ПЕРЕИНДЕКСАЦИЯ ПО feature_names_
            X = X.reindex(columns=list(self.scaler.feature_names_in_))

            X_scaled = self.scaler.transform(X.iloc[[-1]])
            prob = self.model.predict_proba(X_scaled)[:, 1][0]
            return max(0.0, min(1.0, float(prob)))
        except Exception as e:
            logger.error(f'Error in ML predict: {e}')
            return None


# здесь могут быть остальные функции анализа, например analyze_asset()

def analyze_asset(ticker: str, horizon: str, strategy: str):
    client = PolygonClient()
    df = client.daily_ohlc(ticker, days=720)
    ml_model = MLModel()

    price = client.last_trade_price(ticker)
    levels = {}  # заполняем уровни тоже как положено

    confidence = ml_model.predict(df)  # вероятности AI

    # Пример возврата
    return {
        'last_price': price,
        'levels': levels,
        'confidence_breakdown': {
            'ai_probability': confidence
            # остальные показатели / правила
        } if confidence is not None else None
    }

