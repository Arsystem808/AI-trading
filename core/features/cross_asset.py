# core/features/cross_asset.py
def rolling_corr(a: pd.Series, b: pd.Series, win: int = 60) -> pd.Series:
    return a.rolling(win).corr(b)

def vix_regime(vix_series: pd.Series):
    return pd.cut(vix_series, bins=[-np.inf,20,30,np.inf], labels=["low","elevated","crisis"])
