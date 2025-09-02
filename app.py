import os
import datetime as dt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.data import fetch_polygon_ohlc, date_range_days
from core.strategy import analyze, HORIZONS
from core.textgen import build_narrative

load_dotenv()

st.set_page_config(page_title="CapinteL-Q — MVP", page_icon="📈", layout="centered")

st.title("CapinteL‑Q — трейд‑ИИ (MVP)")
st.caption("Говорит, как опытный трейдер. Без раскрытия внутренней методологии.")

with st.sidebar:
    st.header("Параметры")
    symbol = st.text_input("Тикер (Polygon, например AAPL, TSLA, BTCUSD):", "AAPL").upper().strip()
    horizon_key = st.radio("Горизонт анализа:", options=list(HORIZONS.keys()), format_func=lambda x: HORIZONS[x].name, index=0)
    days_back = st.slider("Сколько дней истории загрузить:", 180, 1200, 540, step=30)
    run_btn = st.button("Проанализировать", type="primary")
    st.markdown("---")
    show_dev = st.toggle("⚙️ Dev-панель (только для тестов)", value=False, help="Служебные метрики. Не показывать клиентам.")
    st.caption("Источник данных: Polygon.io")

if run_btn:
    try:
        start, end = date_range_days(days_back)
        with st.spinner("Загружаю данные с Polygon…"):
            df = fetch_polygon_ohlc(symbol, start=start, end=end, timespan="day", multiplier=1)
        st.success(f"Данные получены: {len(df)} баров (дневных).")
        
        res = analyze(symbol, df, horizon_key=horizon_key)
        txt = build_narrative(symbol, HORIZONS[horizon_key].name, res)
        st.markdown(txt)
        
        if show_dev:
            with st.expander("Показать служебные метрики (DEV)"):
                st.json(res["debug"])
                st.dataframe(df.tail(30))
    except Exception as e:
        st.error(str(e))
        st.info("Проверьте тикер и наличие POLYGON_API_KEY в .env")
else:
    st.info("Введите тикер, выберите горизонт и нажмите «Проанализировать».")
