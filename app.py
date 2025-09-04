# app.py
import os
import re
import hashlib
import random
import streamlit as st
from dotenv import load_dotenv
from core.strategy import analyze_asset

# ---------- опциональный импорт OpenAI ----------
def _get_openai_client():
    try:
        from openai import OpenAI  # современный SDK (v1)
    except Exception:
        return None
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key) if key else None

def _llm_chat(system: str, user: str, temperature=0.2, max_tokens=320):
    client = _get_openai_client()
    if not client:
        return None
    try:
        model = st.secrets.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        res = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (res.choices[0].message.content or "").strip()
    except Exception:
        return None

# ---------- базовая настройка ----------
load_dotenv()
st.set_page_config(
    page_title="Arxora — трейд-ИИ (MVP)",
    page_icon="assets/arxora_favicon_512.png",
    layout="centered",
)

# ---------- хедер / брендинг ----------
def render_arxora_header():
    hero_path = "assets/arxora_logo_hero.png"
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)
    else:
        PURPLE = "#5B5BF7"; BLACK = "#0B0D0E"
        st.markdown(
            f"""
            <div style="border-radius:8px;overflow:hidden;
                        box-shadow:0 0 0 1px rgba(0,0,0,.06),0 12px 32px rgba(0,0,0,.18);">
              <div style="background:{PURPLE};padding:28px 16px;">
                <div style="max-width:1120px;margin:0 auto;">
                  <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
                              color:#fff;font-weight:700;letter-spacing:.4px;
                              font-size:clamp(36px,7vw,72px);line-height:1.05;">
                    Arxora
                  </div>
                </div>
              </div>
              <div style="background:{BLACK};padding:12px 16px 16px 16px;">
                <div style="max-width:1120px;margin:0 auto;">
                  <div style="color:#fff;font-size:clamp(16px,2.4vw,28px);opacity:.92;">trade smarter.</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def card_html(title, value, sub=None, color=None):
    bg = "#141a20"
    if color == "green": bg = "#123b2a"
    elif color == "red": bg = "#3b1f20"
    return f"""
        <div style="background:{bg}; padding:12px 16px; border-radius:14px; border:1px solid rgba(255,255,255,0.06); margin:6px 0;">
            <div style="font-size:0.9rem; opacity:0.85;">{title}</div>
            <div style="font-size:1.4rem; font-weight:700; margin-top:4px;">{value}</div>
            {f"<div style='font-size:0.8rem; opacity:0.7; margin-top:2px;'>{sub}</div>" if sub else ""}
        </div>
    """

def chips_html(items):
    if not items: return ""
    chips = "".join([f"<span style='display:inline-block;padding:4px 8px;border-radius:999px;"
                     f"background:rgba(255,255,255,0.06);margin:2px 6px 2px 0;font-size:0.8rem;'>{st._escape_html(x)}</span>"
                     for x in items])
    return f"<div style='margin-top:6px'>{chips}</div>"

def rr_line_3tp(levels):
    try:
        entry = float(levels["entry"]); sl = float(levels["sl"])
        risk = abs(entry - sl)
        if risk <= 1e-9: return ""
        rr1 = abs(float(levels["tp1"]) - entry) / risk
        rr2 = abs(float(levels["tp2"]) - entry) / risk
        rr3 = abs(float(levels["tp3"]) - entry) / risk
        return f"RR ≈ 1:{rr1:.1f} (TP1) · 1:{rr2:.1f} (TP2) · 1:{rr3:.1f} (TP3)"
    except Exception:
        return ""

def normalize_for_polygon(symbol: str) -> str:
    """Приводим ввод к формату Polygon:
       BTCUSDT/ETHUSD → X:BTCUSD/X:ETHUSD; уже X:/C:/O: — просто чистим хвост."""
    s = (symbol or "").strip().upper().replace(" ", "")
    if not s:
        return s
    if s.startswith(("X:", "C:", "O:")):
        head, tail = s.split(":", 1)
        tail = tail.replace("USDT", "USD").replace("USDC", "USD")
        return f"{head}:{tail}"
    if re.match(r"^[A-Z]{2,10}USD(T|C)?$", s):
        s = s.replace("USDT", "USD").replace("USDC", "USD")
        return f"X:{s}"
    return s

def render_plan_qa(ticker: str, horizon: str, out: dict):
    """Мини-чат «Вопросы по плану». Работает только при наличии OPENAI_API_KEY."""
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Вопросы по плану")

    if "qa" not in st.session_state:
        st.session_state.qa = []

    for role, text in st.session_state.qa:
        with st.chat_message(role):
            st.markdown(text)

    user_q = st.chat_input("Спроси про этот сетап…")
    if not user_q:
        return

    st.session_state.qa.append(("user", user_q))

    lv = out.get("levels", {})
    probs = out.get("probs", {})
    rec = out.get("recommendation", {})
    action = rec.get("action", "WAIT")
    conf = rec.get("confidence", 0)

    context_blob = (
        f"Тикер: {ticker}\n"
        f"Горизонт: {horizon}\n"
        f"Действие: {action} (уверенность {int(float(conf)*100)}%)\n"
        f"Цена: {out.get('last_price', 0):.2f}\n"
        f"Entry: {lv.get('entry', 0):.2f}\n"
        f"SL: {lv.get('sl', 0):.2f}\n"
        f"TP1: {lv.get('tp1', 0):.2f} (p~{int(float(probs.get('tp1',0))*100)}%)\n"
        f"TP2: {lv.get('tp2', 0):.2f} (p~{int(float(probs.get('tp2',0))*100)}%)\n"
        f"TP3: {lv.get('tp3', 0):.2f} (p~{int(float(probs.get('tp3',0))*100)}%)\n"
        f"Контекст: {', '.join(out.get('context', []))}\n"
        "Отвечай, опираясь ТОЛЬКО на эти факты. Не раскрывай внутренние формулы."
    )

    system = (
        "Ты помощник трейд-платформы Arxora. "
        "Отвечай коротко (1–4 предложения), профессионально, без воды и жаргона. "
        "Не давай персональных инвестиционных рекомендаций; объясняй логику плана, риски и альтернативы."
    )

    ans = _llm_chat(system, f"{context_blob}\n\nВопрос пользователя: {user_q}")
    if not ans:
        ans = "AI-ответ отключён (нет OPENAI_API_KEY или библиотека OpenAI не установлена)."

    st.session_state.qa.append(("assistant", ans))
    st.rerun()

# ---------- UI ----------
render_arxora_header()

col1, col2 = st.columns([2, 1])
with col1:
    ticker_input = st.text_input("Тикер", value="", placeholder="Примеры: AAPL · TSLA · X:BTCUSD · BTCUSDT")
    ticker_raw = ticker_input.strip().upper()
with col2:
    horizon = st.selectbox(
        "Горизонт",
        ["Краткосрок (1–5 дней)", "Среднесрок (1–4 недели)", "Долгосрок (1–6 месяцев)"],
        index=1
    )

symbol = normalize_for_polygon(ticker_raw)
run = st.button("Проанализировать", type="primary")

# ---------- Main ----------
if run:
    try:
        out = analyze_asset(ticker=symbol, horizon=horizon)

        # Цена крупно
        st.markdown(
            f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>${out['last_price']:.2f}</div>",
            unsafe_allow_html=True,
        )

        # Рекомендация
        action = out["recommendation"]["action"]
        conf = float(out["recommendation"].get("confidence", 0.0))
        conf_pct = f"{int(round(conf*100))}%"
        action_text = "Buy LONG" if action == "BUY" else ("Sell SHORT" if action == "SHORT" else "WAIT")

        st.markdown(
            f"""
            <div style="background:#0f1b2b; padding:14px 16px; border-radius:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:10px;">
                <div style="font-size:1.15rem; font-weight:700;">{action_text}</div>
                <div style="opacity:0.75; font-size:0.95rem; margin-top:2px;">{conf_pct} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        lv = out["levels"]

        # Уровни для BUY/SHORT
        if action in ("BUY", "SHORT"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(card_html("Entry", f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2:
                st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(card_html("TP 1", f"{lv['tp1']:.2f}", sub=f"Probability {int(round(out['probs'].get('tp1',0)*100))}%"),
                            unsafe_allow_html=True)
            with c2:
                st.markdown(card_html("TP 2", f"{lv['tp2']:.2f}", sub=f"Probability {int(round(out['probs'].get('tp2',0)*100))}%"),
                            unsafe_allow_html=True)
            with c3:
                st.markdown(card_html("TP 3", f"{lv['tp3']:.2f}", sub=f"Probability {int(round(out['probs'].get('tp3',0)*100))}%"),
                            unsafe_allow_html=True)

            rr = rr_line_3tp(lv)
            if rr:
                st.markdown(f"<div style='opacity:0.75; margin-top:4px'>{rr}</div>", unsafe_allow_html=True)

        # Контекст-чипсы
        st.markdown(chips_html(out.get("context", [])), unsafe_allow_html=True)

        # Живой текст из стратегии (если есть)
        if out.get("note_html"):
            st.markdown(out["note_html"], unsafe_allow_html=True)

        # Альтернативный сценарий
        if out.get("alt"):
            st.markdown(
                f"<div style='margin-top:6px;'><b>Если пойдёт против базового сценария:</b> {out['alt']}</div>",
                unsafe_allow_html=True,
            )

        # Дисклеймер
        st.caption("Данная информация является примером того, как AI может генерировать инвестиционные идеи и не является прямой инвестиционной рекомендацией. Торговля на финансовых рынках сопряжена с высоким риском.")

        # Блок Q&A (работает при наличии OPENAI_API_KEY)
        render_plan_qa(ticker_raw or symbol, horizon, out)

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
else:
    st.info("Введите тикер и нажмите «Проанализировать».")
