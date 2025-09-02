import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from core.strategy import analyze_asset

st.set_page_config(page_title="CapinteL‑Q — трейд‑ИИ (MVP)", page_icon="📈", layout="centered")

st.markdown("# CapinteL‑Q — трейд‑ИИ (MVP)")
st.caption("Говорит, как опытный трейдер. Без раскрытия внутренней методологии.")

# --- Inputs ---
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Тикер", value="AAPL").strip().upper()
with col2:
    horizon = st.selectbox(
        "Горизонт",
        ["Краткосрок (1–5 дней)", "Среднесрок (1–4 недели)", "Долгосрок (1–6 месяцев)"],
        index=1
    )

run = st.button("Проанализировать", type="primary")

if run:
    try:
        result = analyze_asset(ticker=ticker, horizon=horizon)
        st.subheader(f"{ticker} — {horizon}")
        if result.get("background"):
            st.write("**Фон:** " + result["background"])
        # Recommendation block (no confidence scale, no bars mentions)
        rec = result.get("recommendation", {})
        action = rec.get("action", "WAIT")
        st.write(f"**Рекомендация:** {action}")
        # Levels
        lv = result.get("levels", {})
        if lv:
            st.write(
                f"**План:** вход {lv.get('entry_str','—')}; "
                f"цели {lv.get('tp1_str','—')}" + (f", {lv.get('tp2_str','—')}" if lv.get('tp2_str') else "") +
                f"; стоп {lv.get('sl_str','—')}."
            )

        # Alternative scenario (no word 'Альт', no repeated labels)
        alt = result.get("alternative")
        if alt:
            st.write(f"**Если пойдёт против базового сценария:** {alt}")

        comment = result.get("comment")
        if comment:
            st.write("**Комментарий:** " + comment)

        st.caption("Это не инвестиционная рекомендация. Решения вы принимаете самостоятельно.")
    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
        st.stop()
else:
    st.info("Введите тикер и нажмите «Проанализировать».")
