import os, sys, json
import numpy as np, pandas as pd
from pathlib import Path
if os.getenv("CI_DRY_RUN") == "1":
    print("CI_DRY_RUN=1 -> skip train_m7")
    sys.exit(0)

def load_data():
    # если нет API-ключа, синтетика
    n=1000
    idx = pd.date_range(periods=n, end=pd.Timestamp.utcnow(), freq="B")
    base = np.linspace(100, 120, n) + np.sin(np.linspace(0,25,n))*3
    df = pd.DataFrame({
        "open":  base + 0.1,
        "high":  base + 0.8,
        "low":   base - 0.8,
        "close": base
    }, index=idx)
    return df

def make_features(df: pd.DataFrame):
    x = pd.DataFrame({
        "ret1": df["close"].pct_change().fillna(0),
        "ret5": df["close"].pct_change(5).fillna(0),
        "vol5": df["close"].pct_change().rolling(5).std().fillna(0),
        "slope": pd.Series(df["close"]).rolling(10).apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0]).fillna(0)
    }, index=df.index)
    y = (df["close"].shift(-5) > df["close"]).astype(int).fillna(0)
    x, y = x.iloc[:-5], y.iloc[:-5]
    return x.values, y.values

def train(x,y):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=-1)
    clf.fit(x,y)
    return clf

def main():
    df = load_data()
    x,y = make_features(df)
    model = train(x,y)
    Path("models").mkdir(exist_ok=True)
    import joblib; joblib.dump(model, "models/m7model.pkl")
    print("saved models/m7model.pkl")
if __name__ == "__main__":
    main()
