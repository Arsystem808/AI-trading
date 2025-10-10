# core/features/technical.py
import numpy as np
import pandas as pd

def detect_regime(df: pd.DataFrame) -> pd.Series:
    # простая эвристика: волатильность и наклон
    vol = df['close'].pct_change().rolling(50).std()
    slope = df['close'].pct_change().rolling(50).mean()
    regime = np.select(
        [vol<vol.quantile(0.4), slope>0],
        ["ranging","trending"],
        default="volatile"
    )
    return pd.Series(regime, index=df.index)
