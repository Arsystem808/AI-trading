# core/etl/pipeline.py
import pandas as pd
from core.etl.validate import run_gx_checks
from core.monitoring.drift_evidently import drift_report
from core.feature_store.io import get_store
from core.features.registry import FeatureRegistry
from core.features.technical import detect_regime
from core.features.microstructure import bid_ask_imbalance, vpin
from core.features.cross_asset import rolling_corr, vix_regime

def run_etl(market_df: pd.DataFrame, btc: pd.Series, vix: pd.Series, l1: pd.DataFrame, trades: pd.DataFrame):
    # базовые
    market_df["regime"] = detect_regime(market_df)
    market_df["obi"] = bid_ask_imbalance(l1).reindex(market_df.index).ffill()
    market_df["vpin"] = vpin(trades)
    market_df["btc_corr_60"] = rolling_corr(market_df["returns"], btc, 60)
    market_df["vix_regime"] = vix_regime(vix).reindex(market_df.index).ffill()

    assert run_gx_checks(market_df), "GX data quality failed"

    # записать снапшот и в Feast offline
    out = "data/features/latest.parquet"
    market_df.to_parquet(out, index=True)
    store = get_store()
    # дальнейшая загрузка согласно источникам store (пример оставлен абстрактным)

    # отчет о дрейфе
    # ref_df должен храниться (например, вчерашний/недельный эталон)
    # drift_report(ref_df, market_df)..

    return out
