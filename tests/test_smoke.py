import pandas as pd, numpy as np
import datetime as dt

def _fake_df(n=240):
    idx = pd.date_range(end=dt.datetime.utcnow(), periods=n, freq='B')
    base = np.linspace(100, 110, n) + np.sin(np.linspace(0,8,n))*2
    return pd.DataFrame({
        'open': base + 0.1,
        'high': base + 0.6,
        'low':  base - 0.6,
        'close': base
    }, index=idx)

def test_analyzers_offline():
    import core.strategy as S
    class FakePolygon:
        def daily_ohlc(self, ticker, days=120): return _fake_df(max(120, days))
        def last_trade_price(self, ticker): return float(_fake_df().close.iloc[-1])
    S.PolygonClient = FakePolygon
    for strat in ['Global','M7','W7','AlphaPulse','Octopus']:
        res = S.analyze_asset('AAPL','Краткосрочный', strat)
        assert 'recommendation' in res and 'levels' in res and 'probs' in res
