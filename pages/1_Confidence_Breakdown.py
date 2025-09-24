import os, sys, json, subprocess
from pathlib import Path
import streamlit as st

st.title("Confidence Breakdown")
ticker = st.text_input("Введите тикер", "SPY")
if st.button("Запросить"):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "signal_with_confidence.py"), ticker],
        cwd=str(repo_root), env=env, capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        st.error(proc.stderr.strip())
    else:
        data = json.loads(proc.stdout)
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
