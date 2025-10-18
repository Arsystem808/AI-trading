# ui/performance_viewer.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

METRICS_CSV = Path("metrics/agent_performance.csv")


def load_agent_signals(agent: str, ticker: str, days: int = 90) -> pd.DataFrame:
    """Загрузка сигналов из metrics/agent_performance.csv"""
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(METRICS_CSV)
        
        # Фильтр по агенту (case-insensitive)
        if agent:
            df = df[df['agent'].str.contains(agent, case=False, na=False)]
        
        # Фильтр по тикеру
        if ticker:
            df = df[df['ticker'] == ticker]
        
        # Фильтр по времени
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['ts'] >= cutoff]
        
        return df
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return pd.DataFrame()


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Расчёт метрик из сигналов"""
    if df.empty:
        return {"signal_count": 0, "win_rate": 0.0, "avg_confidence": 0.0}
    
    total = len(df)
    wins = len(df[df['confidence'] > 0.65])
    win_rate = wins / total if total > 0 else 0.0
    
    return {
        "signal_count": total,
        "win_rate": win_rate,
        "avg_confidence": df['confidence'].mean() if 'confidence' in df.columns else 0.0
    }


def plot_performance_chart(agent: str, ticker: str):
    """График производительности агента"""
    df = load_agent_signals(agent, ticker, days=90)
    
    if df.empty:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1e3a5f 0%, #2a5780 100%);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            color: #94a3b8;
            border: 1px solid #334155;
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">📊</div>
            <div style="font-size: 18px; font-weight: 500;">Данных пока нет</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Агрегация по дням
    df['date'] = df['ts'].dt.date
    daily = df.groupby('date').agg({
        'confidence': 'mean',
        'ts': 'count'
    }).reset_index()
    daily.columns = ['date', 'avg_confidence', 'signal_count']
    
    # График
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['avg_confidence'],
        mode='lines+markers',
        name='Уверенность',
        line=dict(color='#10b981', width=3),
        marker=dict(size=6, color='#10b981'),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)',
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.0%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=None,
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1],
            gridcolor='#334155',
            showgrid=True
        ),
        xaxis=dict(
            gridcolor='#334155',
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', size=12),
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Метрики
    metrics = calculate_metrics(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Сигналов", metrics["signal_count"])
    with col2:
        st.metric("🎯 Win Rate", f"{metrics['win_rate']:.0%}")
    with col3:
        st.metric("💪 Avg Confidence", f"{metrics['avg_confidence']:.0%}")


def render_performance_section(model: str):
    """Основной рендер блока эффективности"""
    st.subheader(f"Эффективность модели {model} по ключевым инструментам (3 месяца)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### SPY")
        plot_performance_chart(model, "SPY")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### BTCUSD")
        plot_performance_chart(model, "X:BTCUSD")
    
    with col2:
        st.markdown("### QQQ")
        plot_performance_chart(model, "QQQ")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### ETHUSD")
        plot_performance_chart(model, "X:ETHUSD")
