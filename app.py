# app.py
# Octopus — График цены + История (Rules / AI override)
# Профессиональный вариант "Timeline" с точками над/под свечами и насыщенностью цвета = confidence.
# Не содержит блоков "эффективности моделей".

from __future__ import annotations

import os
import io
import json
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Попробуем импортировать yfinance для OHLC; если нет — подскажем пользователю.
try:
    import yfinance as yf  # type: ignore
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


# ----------------------------
# Конфигурация страницы
# ----------------------------
st.set_page_config(
    page_title="Octopus — График + История",
    layout="wide",
    page_icon="🧠",
)

# Палитры для точек
BLUES = "Blues"       # для базовых правил
PURPLES = "Purples"   # для AI override

# ----------------------------
# Модель данных событий
# ----------------------------
Kind = Literal["rules", "override"]

@dataclass
class EventPoint:
    ts: pd.Timestamp          # timestamp в UTC
    kind: Kind                # "rules" | "override"
    confidence: float         # 0..100
    side: Optional[Literal["long", "short"]] = None  # необязательно


# ----------------------------
# Утилиты
# ----------------------------
def _to_utc(ts: pd.Timestamp | datetime | str) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        t = ts
    elif isinstance(ts, datetime):
        t = pd.Timestamp(ts)
    else:
        t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def load_ohlc(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Загружает OHLC в UTC. Для стабильности использует yfinance.
    Возвращает DataFrame с колонками: Open, High, Low, Close, Volume
    """
    if not YF_AVAILABLE:
        st.error("Модуль yfinance недоступен. Установите пакет: pip install yfinance")
        return pd.DataFrame()

    yf_symbol = symbol
    # Примеры тикеров: BTC-USD, ETH-USD, AAPL
    df = yf.download(
        yf_symbol,
        start=start,
        end=end + timedelta(days=1),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df.empty:
        return df

    # У yfinance индекс tz-наивный в локали Нью-Йорк; приводим к UTC.
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )
    return df[["Open", "High", "Low", "Close", "Volume"]]


def parse_events_from_json_bytes(
    content: bytes
) -> List[EventPoint]:
    """
    Ожидается список объектов:
    {
      "ts": "2025-10-01T12:30:00Z",
      "kind": "rules" | "override",
      "confidence": 63.0,
      "side": "long" | "short" | null
    }
    """
    payload = json.loads(content.decode("utf-8"))
    events: List[EventPoint] = []
    if isinstance(payload, dict) and "events" in payload:
        payload = payload["events"]
    if not isinstance(payload, list):
        raise ValueError("JSON должен быть массивом событий или объектом с ключом 'events'")

    for item in payload:
        ts = _to_utc(item["ts"])
        kind = str(item["kind"]).lower()
        if kind not in ("rules", "override"):
            continue
        confidence = float(item.get("confidence", 0.0))
        side = item.get("side")
        if side is not None:
            side = str(side).lower()
            if side not in ("long", "short"):
                side = None
        events.append(EventPoint(ts=ts, kind=kind, confidence=confidence, side=side))
    return events


def demo_events_from_ohlc(
    df: pd.DataFrame,
    seed: int = 7,
    p_rules: float = 0.06,
    p_override: float = 0.04,
) -> List[EventPoint]:
    """
    Генерирует демо-события поверх OHLC, чтобы сразу увидеть результат.
    """
    if df.empty:
        return []
    rng = np.random.default_rng(seed)
    idx = df.index

    mask_rules = rng.random(len(idx)) < p_rules
    mask_override = rng.random(len(idx)) < p_override

    events: List[EventPoint] = []
    for ts, flag in zip(idx, mask_rules):
        if flag:
            events.append(
                EventPoint(
                    ts=_to_utc(ts),
                    kind="rules",
                    confidence=float(np.clip(rng.normal(65, 15), 5, 99)),
                    side=rng.choice(["long", "short"]),
                )
            )
    for ts, flag in zip(idx, mask_override):
        if flag:
            events.append(
                EventPoint(
                    ts=_to_utc(ts),
                    kind="override",
                    confidence=float(np.clip(rng.normal(55, 20), 5, 99)),
                    side=rng.choice(["long", "short"]),
                )
            )
    return events


def _align_events_to_candles(
    events: List[EventPoint], df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает события на два DataFrame (rules, override) и сопоставляет им Y-координаты
    относительно свечей (над High и под Low).
    """
    if df.empty or not events:
        return pd.DataFrame(), pd.DataFrame()

    ts_to_row = pd.Series(range(len(df)), index=df.index)

    # Подготовка таблиц
    recs_rules = []
    recs_over = []

    for e in events:
        # Находим близжайшую свечу по времени (floor к индексу)
        # Берем индекс свечи, чей ts <= e.ts
        ts = e.ts
        # Если точного индекса нет, берем последнюю свечу не позже ts
        pos = df.index.get_indexer([ts], method="nearest")
        if len(pos) == 0 or pos[0] < 0:
            continue
        i = int(pos[0])
        i = min(max(i, 0), len(df) - 1)
        row_ts = df.index[i]
        high = float(df.iloc[i]["High"])
        low = float(df.iloc[i]["Low"])
        mid = (high + low) / 2.0
        atr = max(high - low, 1e-9)

        if e.kind == "rules":
            # точка над свечой
            y = high + atr * 0.06
            recs_rules.append(
                dict(
                    ts=row_ts,
                    y=y,
                    confidence=float(e.confidence),
                    side=e.side or "",
                )
            )
        else:
            # точка под свечой
            y = low - atr * 0.06
            recs_over.append(
                dict(
                    ts=row_ts,
                    y=y,
                    confidence=float(e.confidence),
                    side=e.side or "",
                )
            )

    rules_df = pd.DataFrame.from_records(recs_rules)
    over_df = pd.DataFrame.from_records(recs_over)
    return rules_df, over_df


def build_chart(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    over_df: pd.DataFrame,
    show_rules: bool,
    show_override: bool,
    min_conf: float,
    title: str,
) -> go.Figure:
    """
    Собирает candlestick + 2 Scatter‑слоя точек с colorscale по confidence.
    """
    fig = go.Figure()

    # Свечи
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Цена",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=True,
        )
    )

    # Фильтры по порогу уверенности
    if show_rules and not rules_df.empty:
        rr = rules_df[rules_df["confidence"] >= float(min_conf)].copy()
        if not rr.empty:
            fig.add_trace(
                go.Scatter(
                    x=rr["ts"],
                    y=rr["y"],
                    mode="markers",
                    name="Базовые правила",
                    marker=dict(
                        size=np.clip((rr["confidence"] / 100.0) * 14 + 6, 6, 20),
                        color=rr["confidence"],
                        colorscale=BLUES,
                        cmin=0,
                        cmax=100,
                        line=dict(color="rgba(0,0,0,0.0)", width=0),
                        opacity=0.95,
                        symbol="circle",
                    ),
                    hovertemplate=(
                        "<b>Rules</b><br>"
                        "Время: %{x}<br>"
                        "Уверенность: %{marker.color:.1f}%<br>"
                        "Сторона: %{text}<extra></extra>"
                    ),
                    text=rr["side"].fillna(""),
                    showlegend=True,
                )
            )

    if show_override and not over_df.empty:
        oo = over_df[over_df["confidence"] >= float(min_conf)].copy()
        if not oo.empty:
            fig.add_trace(
                go.Scatter(
                    x=oo["ts"],
                    y=oo["y"],
                    mode="markers",
                    name="AI override",
                    marker=dict(
                        size=np.clip((oo["confidence"] / 100.0) * 14 + 6, 6, 20),
                        color=oo["confidence"],
                        colorscale=PURPLES,
                        cmin=0,
                        cmax=100,
                        line=dict(color="rgba(0,0,0,0.0)", width=0),
                        opacity=0.95,
                        symbol="circle",
                    ),
                    hovertemplate=(
                        "<b>AI override</b><br>"
                        "Время: %{x}<br>"
                        "Уверенность: %{marker.color:.1f}%<br>"
                        "Сторона: %{text}<extra></extra>"
                    ),
                    text=oo["side"].fillna(""),
                    showlegend=True,
                )
            )

    # Оформление
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(text=title, x=0.01, xanchor="left"),
        xaxis=dict(
            showgrid=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        hovermode="x unified",
    )
    return fig


# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.subheader("Настройки")
    default_symbol = "BTC-USD"
    symbol = st.text_input("Тикер", value=default_symbol, help="Пример: BTC-USD, ETH-USD, AAPL")
    today = datetime.now(timezone.utc).date()
    start_date: date = st.date_input("Начало", value=today - timedelta(days=30))
    end_date: date = st.date_input("Конец", value=today)
    interval = st.selectbox("Интервал", ["15m", "30m", "1h", "4h", "1d"], index=2)

    st.markdown("---")
    st.caption("События для таймлайна")
    source_mode = st.radio(
        "Источник",
        ["Демо (сгенерировать)", "Загрузить JSON"],
        index=0,
        horizontal=False,
    )
    min_conf = st.slider("Минимальная уверенность, %", 0, 100, 40, step=1)
    show_rules = st.toggle("Показывать базовые правила", value=True)
    show_override = st.toggle("Показывать AI override", value=True)

    st.markdown("---")
    st.caption("Экспорт")
    want_html = st.toggle("Скачать график как HTML", value=False)

st.title("График цены + История (Rules / AI override)")

# Загрузка OHLC
start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)

with st.spinner("Загружаем данные цены..."):
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
            events = parse_events_from_json_bytes(uploaded.read())
            st.success(f"Загружено событий: {len(events)}")
        except Exception as e:
            st.error(f"Ошибка парсинга JSON: {e}")
            events = []

# Привязка событий к свечам
rules_df, over_df = _align_events_to_candles(events, ohlc)

# Сборка графика
title = f"{symbol} — {interval} • Таймлайн событий"
fig = build_chart(
    df=ohlc,
    rules_df=rules_df,
    over_df=over_df,
    show_rules=show_rules,
    show_override=show_override,
    min_conf=float(min_conf),
    title=title,
)

st.plotly_chart(fig, use_container_width=True)

# Экспорт (опционально HTML; без внешних зависимостей)
if want_html:
    html_bytes = fig.to_html(full_html=True).encode("utf-8")
    st.download_button(
        "Скачать HTML",
        data=html_bytes,
        file_name=f"{symbol.replace('/', '-')}_{interval}_timeline.html",
        mime="text/html",
    )

# Подсказка по формату JSON
with st.expander("Формат JSON для событий (пример)"):
    example = [
        {
            "ts": (pd.Timestamp.utcnow() - pd.Timedelta("1h")).isoformat(),
            "kind": "rules",
            "confidence": 62.5,
            "side": "long",
        },
        {
            "ts": (pd.Timestamp.utcnow() - pd.Timedelta("2h")).isoformat(),
            "kind": "override",
            "confidence": 48.0,
            "side": "short",
        },
    ]
    st.code(json.dumps(example, ensure_ascii=False, indent=2), language="json")
