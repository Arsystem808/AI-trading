# app.py — График свечей + История (Rules / AI override)
# Приложение автоматически устанавливает недостающие пакеты (plotly, yfinance)
# и не содержит секций "эффективности моделей". Красивый таймлайн поверх свечей.

from __future__ import annotations

import sys
import subprocess
import tempfile
import pathlib
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Автоустановка недостающих пакетов (один раз на запуск) ----------
def ensure_pkg(pkg: str, import_name: Optional[str] = None, version: Optional[str] = None):
    try:
        __import__(import_name or pkg)
        return
    except ModuleNotFoundError:
        pass
    lock = pathlib.Path(tempfile.gettempdir()) / f".install_{pkg}.lock"
    try:
        if not lock.exists():
            lock.write_text("lock")
            spec = pkg if version is None else f"{pkg}=={version}"
            subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
    finally:
        if lock.exists():
            lock.unlink(missing_ok=True)
    __import__(import_name or pkg)

# Обязательные для графика зависимости
ensure_pkg("plotly", "plotly")        # требуется для st.plotly_chart [docs] [web:136]
ensure_pkg("yfinance", "yfinance")    # для OHLC‑данных [пример использования в UI] [web:133]

import plotly.graph_objects as go      # после ensure_pkg импорт пройдёт [web:133]
import yfinance as yf                  # загрузка OHLC [web:133]

# ---------------------------- Конфиг страницы ----------------------------
st.set_page_config(
    page_title="Octopus — Timeline на графике",
    layout="wide",
    page_icon="🧠",
)

# Цветовые шкалы для точек
BLUES = "Blues"       # базовые правила
PURPLES = "Purples"   # AI override

Kind = Literal["rules", "override"]

@dataclass
class EventPoint:
    ts: pd.Timestamp
    kind: Kind
    confidence: float      # 0..100
    side: Optional[Literal["long", "short"]] = None

def _to_utc(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts, utc=True)
    return t

def load_ohlc(symbol: str, start: datetime, end: datetime, interval: str = "1h") -> pd.DataFrame:
    # yfinance возвращает индекс без tz; приводим к UTC для корректной привязки точек [web:133]
    df = yf.download(
        symbol,
        start=start,
        end=end + timedelta(days=1),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]

def parse_events_json_bytes(content: bytes) -> List[EventPoint]:
    import json
    payload = json.loads(content.decode("utf-8"))
    if isinstance(payload, dict) and "events" in payload:
        payload = payload["events"]
    out: List[EventPoint] = []
    for it in payload:
        ts = _to_utc(it["ts"])
        kind = str(it["kind"]).lower()
        if kind not in ("rules", "override"):
            continue
        conf = float(it.get("confidence", 0.0))
        side = it.get("side")
        side = (str(side).lower() if side else None)
        if side not in (None, "long", "short"):
            side = None
        out.append(EventPoint(ts=ts, kind=kind, confidence=conf, side=side))
    return out

def demo_events_from_ohlc(df: pd.DataFrame, seed: int = 42) -> List[EventPoint]:
    if df.empty:
        return []
    rng = np.random.default_rng(seed)
    idx = df.index

    rules_mask = rng.random(len(idx)) < 0.06
    over_mask  = rng.random(len(idx)) < 0.04

    evs: List[EventPoint] = []
    for ts, f in zip(idx, rules_mask):
        if f:
            evs.append(EventPoint(_to_utc(ts), "rules",
                                  float(np.clip(rng.normal(65, 15), 5, 99)),
                                  rng.choice(["long", "short"])))
    for ts, f in zip(idx, over_mask):
        if f:
            evs.append(EventPoint(_to_utc(ts), "override",
                                  float(np.clip(rng.normal(55, 20), 5, 99)),
                                  rng.choice(["long", "short"])))
    return evs

def _align_events_to_candles(events: List[EventPoint], df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or not events:
        return pd.DataFrame(), pd.DataFrame()
    rec_rules, rec_over = [], []
    for ts in sorted(set([e.ts for e in events])):
        pass  # просто принудительная сортировка по времени (визуально стабильнее)

    for e in events:
        pos = df.index.get_indexer([e.ts], method="nearest")
        if len(pos) == 0 or pos[0] < 0:
            continue
        i = int(pos[0])
        row = df.iloc[i]
        atr = max(float(row["High"] - row["Low"]), 1e-9)
        if e.kind == "rules":
            rec_rules.append(dict(ts=df.index[i], y=float(row["High"] + atr * 0.06),
                                  confidence=float(e.confidence), side=e.side or ""))
        else:
            rec_over.append(dict(ts=df.index[i], y=float(row["Low"]  - atr * 0.06),
                                 confidence=float(e.confidence), side=e.side or ""))

    return pd.DataFrame(rec_rules), pd.DataFrame(rec_over)

def build_chart(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    over_df: pd.DataFrame,
    show_rules: bool,
    show_override: bool,
    min_conf: float,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    # Свечи
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Цена",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=True,
        )
    )
    # Точки: Rules
    if show_rules and not rules_df.empty:
        rr = rules_df[rules_df["confidence"] >= float(min_conf)]
        if not rr.empty:
            fig.add_trace(
                go.Scatter(
                    x=rr["ts"], y=rr["y"], mode="markers", name="Базовые правила",
                    marker=dict(
                        size=np.clip((rr["confidence"]/100.0)*14 + 6, 6, 20),
                        color=rr["confidence"], colorscale="Blues", cmin=0, cmax=100,
                        opacity=0.95, symbol="circle"),
                    text=rr["side"].fillna(""),
                    hovertemplate="<b>Rules</b><br>Время: %{x}<br>Уверенность: %{marker.color:.1f}%<br>Сторона: %{text}<extra></extra>",
                    showlegend=True,
                )
            )
    # Точки: Override
    if show_override and not over_df.empty:
        oo = over_df[over_df["confidence"] >= float(min_conf)]
        if not oo.empty:
            fig.add_trace(
                go.Scatter(
                    x=oo["ts"], y=oo["y"], mode="markers", name="AI override",
                    marker=dict(
                        size=np.clip((oo["confidence"]/100.0)*14 + 6, 6, 20),
                        color=oo["confidence"], colorscale="Purples", cmin=0, cmax=100,
                        opacity=0.95, symbol="circle"),
                    text=oo["side"].fillna(""),
                    hovertemplate="<b>AI override</b><br>Время: %{x}<br>Уверенность: %{marker.color:.1f}%<br>Сторона: %{text}<extra></extra>",
                    showlegend=True,
                )
            )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        height=720,
        title=dict(text=title, x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=False), showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        hovermode="x unified",
    )
    return fig

# ---------------------------- UI ----------------------------
with st.sidebar:
    st.subheader("Настройки")
    symbol = st.text_input("Тикер", value="BTC-USD", help="Пример: BTC-USD, ETH-USD, AAPL")
    today = datetime.now(timezone.utc).date()
    start_date: date = st.date_input("Начало", value=today - timedelta(days=30))
    end_date: date = st.date_input("Конец", value=today)
    interval = st.selectbox("Интервал", ["15m", "30m", "1h", "4h", "1d"], index=2)

    st.markdown("---")
    st.caption("События для таймлайна")
    source_mode = st.radio("Источник", ["Демо (сгенерировать)", "Загрузить JSON"], index=0)
    min_conf = st.slider("Минимальная уверенность, %", 0, 100, 40, step=1)
    show_rules = st.toggle("Показывать базовые правила", value=True)
    show_override = st.toggle("Показывать AI override", value=True)

st.title("График цены + Таймлайн событий")

# Данные цены
start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)

with st.spinner("Загружаем цену..."):
    ohlc = load_ohlc(symbol, start=start_dt, end=end_dt, interval=interval)

if ohlc.empty:
    st.warning("Нет данных для выбранного диапазона/тикера.")
    st.stop()

# События
events: List[EventPoint] = []
if source_mode.startswith("Демо"):
    events = demo_events_from_ohlc(ohlc)
else:
    uploaded = st.file_uploader("Загрузите JSON со списком событий", type=["json"])
    if uploaded is not None:
        try:
            events = parse_events_json_bytes(uploaded.read())
            st.success(f"Загружено событий: {len(events)}")
        except Exception as e:
            st.error(f"Ошибка парсинга JSON: {e}")
            events = []

rules_df, over_df = _align_events_to_candles(events, ohlc)

title = f"{symbol} — {interval} • Timeline (Rules / AI override)"
fig = build_chart(
    df=ohlc,
    rules_df=rules_df,
    over_df=over_df,
    show_rules=show_rules,
    show_override=show_override,
    min_conf=float(min_conf),
    title=title,
)

# Визуализация plotly в Streamlit
st.plotly_chart(fig, use_container_width=True)  # встроенная поддержка plotly в Streamlit [web:133]

