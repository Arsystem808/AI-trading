# core/features/microstructure.py
import numpy as np
import pandas as pd

def bid_ask_imbalance(level1: pd.DataFrame) -> pd.Series:
    b = level1['bid_size']
    a = level1['ask_size']
    return (b - a) / (b + a).replace(0, np.nan)

def vpin(trades: pd.DataFrame, buckets: int = 50):
    # trades: columns ['price','size','side'] with side in {+1 buy, -1 sell}
    V = trades['size'].sum() / buckets
    cumv, curv, imbalances = 0.0, 0.0, []
    buyv = sellv = 0.0
    for _, r in trades.iterrows():
        curv += r['size']
        if r['side'] > 0: buyv += r['size']
        else: sellv += r['size']
        if curv >= V:
            imbalances.append(abs(buyv - sellv))
            curv = 0.0; buyv = sellv = 0.0
    if not imbalances: 
        return np.nan
    return np.sum(imbalances) / (len(imbalances) * V)
