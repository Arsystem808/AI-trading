import os
import streamlit as st
from core.strategy import analyze_asset
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="CapinteL-Q — трейд-ИИ (MVP)", page_icon="📈", layout="centered")

# Header (no tagline requested)
st.markdown("<h1 style='margin-bottom:0.2rem;'>CapinteL‑Q — трейд‑ИИ (MVP)</h1>", unsafe_allow_html=True)
st.caption("")  # spacer

# --- Inputs ---
with st.container():
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

def card(title, value, sub=None, color=None):
    bg = "#141a20"
    if color == "green":
        bg = "#123b2a"
    elif color == "red":
        bg = "#3b1f20"
    st.markdown(f"""
    <div style="background:{bg}; padding:12px 16px; border-radius:14px; border:1px solid rgba(255,255,255,0.06); margin:6px 0;">
        <div style="font-size:0.9rem; opacity:0.85;">{title}</div>
        <div style="font-size:1.4rem; font-weight:700; margin-top:4px;">{value}</div>
        {f"<div style='font-size:0.8rem; opacity:0.7; margin-top:2px;'>{sub}</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)

if run:
    try:
        out = analyze_asset(ticker=ticker, horizon=horizon)
        # Top price
        st.markdown(f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>${out['last_price']:.2f}</div>", unsafe_allow_html=True)

        # Action + confidence
        action = out['recommendation']['action']
        conf = out['recommendation']['confidence']
        action_text = "Buy LONG" if action == "BUY" else ("Sell SHORT" if action == "SHORT" else "WAIT")
        st.markdown(f"""
        <div style="background:#0f1b2b; padding:14px 16px; border-radius:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:10px;">
            <div style="font-size:1.15rem; font-weight:700;">{action_text}</div>
            <div style="opacity:0.75; font-size:0.95rem; margin-top:2px;">{int(round(conf*100))}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        lv = out["levels"]
        c1, c2 = st.columns(2)
        with c1:
            card("Entry", f"{lv['entry']:.2f}", color="green")
        with c2:
            card("Stop Loss", f"{lv['sl']:.2f}", color="red")

        c1, c2, c3 = st.columns(3)
        with c1:
            card("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out['probs']['tp1']*100))}%")
        with c2:
            card("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out['probs']['tp2']*100))}%")
        with c3:
            card("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out['probs']['tp3']*100))}%")

        if out.get("note"):
            st.markdown(f"<div style='margin-top:12px; opacity:0.85;'>{out['note']}</div>", unsafe_allow_html=True)

        if out.get("alt"):
            st.markdown(f"<div style='margin-top:8px;'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>", unsafe_allow_html=True)

        st.caption("Это не инвестиционная рекомендация. Решения вы принимаете самостоятельно.")
    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
        if "POLYGON_API_KEY" not in os.environ:
            st.info("Похоже, не задан ключ Polygon. Создайте .env с переменной POLYGON_API_KEY.")
else:
    st.info("Введите тикер и нажмите «Проанализировать».")
