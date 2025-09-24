import streamlit as st
from core.confidence import ui_get_confidence_breakdown

st.title("Confidence Breakdown")

ticker = st.text_input("Введите тикер", "SPY")

if st.button("Запросить"):
    try:
        data = ui_get_confidence_breakdown(ticker)
    except Exception as e:
        st.error(f"Ошибка при получении данных: {e}")
    else:
        st.subheader(f"Сигнал: {data.get('signal','')}")
        st.write(f"Общая уверенность: {data.get('overall_confidence_pct',0):.1f}%")

        b = data.get("breakdown", {})
        st.write(f"— Базовые правила: {b.get('rules_pct',0):.1f}%")
        st.write(f"— AI override: {b.get('ai_override_delta_pct',0):.1f}%")

        shap = data.get("shap_top", [])
        if shap:
            st.write("SHAP факторы влияния:")
            for it in shap:
                st.write(f"{it.get('feature','f')}: {float(it.get('shap',0)):.3f}")
        else:
            st.caption("SHAP факторы недоступны для данного тикера или модели.")
