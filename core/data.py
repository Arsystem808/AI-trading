from __future__ import annotations
import os, datetime as dt
import pandas as pd
import requests
from dotenv import load_dotenv
from typing import Tuple

load_dotenv()

def _read_secret_key() -> str:
    try:
        import streamlit as st
        val = st.secrets.get("POLYGON_API_KEY", "")
        if isinstance(val, str):
            return val.strip()
    except Exception:
        pass
    return ""

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "").strip() or _read_secret_key()

def _polygon_get(url: str, params: dict = None):
    if not POLYGON_KEY:
        raise RuntimeError("Не найден POLYGON_API_KEY (.env или Streamlit Secrets).")
    params = params or {}
    params['apiKey'] = POLYGON_KEY
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Polygon error {resp.status_code}: {resp.text[:300]}")
    return resp.json()

def fetch_polygon_ohlc(ticker: str, start: str, end: str, timespan: str = "day", multiplier: int = 1):
    base = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    data = _polygon_get(base, {"adjusted": "true", "sort": "asc", "limit": 50000})
    results = data.get("results", [])
    if not results:
        raise RuntimeError("Нет данных от Polygon (results пуст).")
    rows = []
    for r in results:
        ts = pd.to_datetime(r['t'], unit='ms')
        rows.append([ts, r['o'], r['h'], r['l'], r['c'], r.get('v', 0.0)])
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).set_index("ts")
    df = df.sort_index()
    return df

def date_range_days(days_back: int = 540) -> Tuple[str, str]:
    end = dt.datetime.utcnow().date()
    start = end - dt.timedelta(days=days_back)
    return start.isoformat(), end.isoformat()

def prev_period_hlc_from_daily(df_daily: pd.DataFrame, period: str = "W") -> Tuple[float,float,float]:
    if df_daily.empty or len(df_daily) < 10:
        raise RuntimeError("Слишком мало данных для расчёта периодов.")
    if period == "W":
        res = df_daily.resample("W-FRI").agg({"high":"max","low":"min","close":"last"})
    elif period == "M":
        res = df_daily.resample("M").agg({"high":"max","low":"min","close":"last"})
    elif period == "Y":
        res = df_daily.resample("Y").agg({"high":"max","low":"min","close":"last"})
    else:
        raise ValueError("period must be W/M/Y")
    if len(res) < 2:
        raise RuntimeError("Недостаточно агрегированных баров для предыдущего периода.")
    prev = res.iloc[-2]
    return float(prev['high']), float(prev['low']), float(prev['close'])
