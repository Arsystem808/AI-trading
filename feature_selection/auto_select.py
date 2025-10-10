# feature_selection/auto_select.py
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

def auto_select_features(X: pd.DataFrame, y: pd.Series, splits: int = 5):
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
    )
    tscv = TimeSeriesSplit(n_splits=splits)
    selector = RFECV(model, step=1, cv=tscv, scoring="roc_auc", n_jobs=-1)
    selector.fit(X, y)
    mask = selector.support_
    return X.columns[mask], getattr(selector.estimator_, "feature_importances_", None)
