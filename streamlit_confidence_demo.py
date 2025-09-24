cat > streamlit_confidence_demo.py <<'PY'
import streamlit as st
import requests

st.title("Confidence Breakdown Demo")
ticker = st.text_input("Введите тикер", "SPY")
if st.button("Запросить"):
    try:
        url = f"http://127.0.0.1:8010/signal?ticker={ticker}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        st.subheader(f"Сигнал: {data.get('signal','')}")
        st.write(f"Общая уверенность: {data.get('overall_confidence_pct',0):.1f}%")
        b = data.get("breakdown",{})
        st.write(f"— Базовые правила: {b.get('rules_pct',0):.1f}%")
        st.write(f"— AI override: {b.get('ai_override_delta_pct',0):.1f}%")
        shap = data.get("shap_top",[])
        if shap:
            st.write("SHAP факторы влияния:")
            for it in shap:
                st.write(f"{it.get('feature','f')}: {float(it.get('shap',0)):.3f}")
    except Exception as e:
        st.error(f"Ошибка при получении данных: {e}")
PY
