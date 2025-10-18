import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from core.strategy import TradingStrategy
import glob
import os

# Настройка страницы
st.set_page_config(
    page_title="Arxora - AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #5865F2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #4752C4;
    }
    h1 {
        color: #5865F2;
        font-weight: 700;
    }
    .metric-card {
        background-color: #1E2130;
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #5865F2;
    }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown("<h1 style='text-align: center; color: #5865F2;'>Arxora</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8E9297;'>trade smarter.</p>", unsafe_allow_html=True)

# Сайдбар
st.sidebar.title("AI agents")
st.sidebar.markdown("**Выберите модель**")

# Модели агентов
model_options = {
    "Octopus": "Orchestrator",
    "AlphaPulse": "Alpha signals",
    "Global": "Global market analysis",
    "M7": "Magnificent 7 stocks",
    "W7": "Week 7 day signals"
}

selected_model = st.sidebar.radio(
    "Модель:",
    options=list(model_options.keys()),
    format_func=lambda x: f"{x} - {model_options[x]}"
)

# Ввод тикера
st.sidebar.markdown("---")
ticker = st.sidebar.text_input("Тикер", value="QQQ", placeholder="Введите тикер...")

# Кнопка анализа
analyze_button = st.sidebar.button("Проанализировать", use_container_width=True)

# Табы
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🤖 AI Insights", "📈 Charts", "⚡ Performance"])

# === ANALYSIS TAB ===
with tab1:
    if analyze_button:
        with st.spinner("🔍 Анализируем рынок..."):
            try:
                # Инициализация стратегии
                strategy = TradingStrategy()
                
                # Получение сигнала
                result = strategy.analyze(ticker, selected_model)
                
                if result and 'signal' in result:
                    st.success(f"✅ Анализ для {ticker} завершён!")
                    
                    # Отображение результатов
                    col1, col2, col3 = st.columns(3)
                    
                    # Сигнал
                    signal_color = {
                        "STRONG BUY": "🟢",
                        "BUY": "🟢",
                        "HOLD": "🟡",
                        "SELL": "🔴",
                        "STRONG SELL": "🔴"
                    }.get(result['signal'], "⚪")
                    
                    with col1:
                        st.metric(
                            label="Signal",
                            value=f"{signal_color} {result['signal']}"
                        )
                    
                    # Confidence
                    with col2:
                        confidence_val = result.get('confidence', 0)
                        st.metric(
                            label="Confidence",
                            value=f"{confidence_val:.1f}%"
                        )
                    
                    # Price
                    with col3:
                        current_price = result.get('current_price', 0)
                        st.metric(
                            label="Current Price",
                            value=f"${current_price:.2f}"
                        )
                    
                    # Breakdown
                    st.markdown("### 📋 Signal Breakdown")
                    
                    if 'breakdown' in result and result['breakdown']:
                        for agent, data in result['breakdown'].items():
                            with st.expander(f"🤖 {agent}"):
                                agent_col1, agent_col2 = st.columns(2)
                                with agent_col1:
                                    st.write(f"**Signal:** {data.get('signal', 'N/A')}")
                                with agent_col2:
                                    st.write(f"**Confidence:** {data.get('confidence', 0):.1f}%")
                                
                                if 'reasoning' in 
                                    st.write(f"**Reasoning:** {data['reasoning']}")
                    else:
                        st.info("No detailed breakdown available")
                else:
                    st.error("❌ Не удалось получить сигнал. Проверьте тикер и попробуйте снова.")
            
            except Exception as e:
                st.error(f"❌ Ошибка при анализе: {str(e)}")
    else:
        st.info("👈 Выберите модель и тикер, затем нажмите **Проанализировать**")

# === AI INSIGHTS TAB ===
with tab2:
    st.header("🤖 AI Insights")
    st.info("Подробная аналитика от AI агентов появится здесь после анализа.")

# === CHARTS TAB ===
with tab3:
    st.header("📈 Charts")
    st.info("Интерактивные графики появятся здесь после анализа.")

# === PERFORMANCE TAB ===
with tab4:
    st.header("📊 Performance Tracking")
    
    # Кнопка пересоздания файла
    if st.button("🔄 Пересобрать performance_summary.csv"):
        perf_files = glob.glob("performance_data/*.csv")
        
        if not perf_files:
            st.warning("⚠️ Нет файлов performance в папке performance_data/")
        else:
            all_data = []
            for file in perf_files:
                try:
                    df = pd.read_csv(file)
                    all_data.append(df)
                except Exception as e:
                    st.error(f"Ошибка чтения {file}: {e}")
            
            if all_
                combined = pd.concat(all_data, ignore_index=True)
                combined.to_csv("performance_summary.csv", index=False)
                st.success("✅ Файл пересобран! Обнови страницу (F5)")
    
    st.markdown("---")
    
    # Загрузка и отображение графиков
    if os.path.exists("performance_summary.csv"):
        try:
            df_perf = pd.read_csv("performance_summary.csv")
            
            if not df_perf.empty and 'timestamp' in df_perf.columns:
                # Преобразуем timestamp
                df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
                
                # Получаем список агентов
                agents = df_perf['agent'].unique() if 'agent' in df_perf.columns else []
                
                if len(agents) > 0:
                    st.subheader("📈 Performance Charts")
                    
                    # График для каждого агента
                    for agent in agents:
                        agent_data = df_perf[df_perf['agent'] == agent].copy()
                        agent_data = agent_data.sort_values('timestamp')
                        
                        # Создаём график
                        fig = go.Figure()
                        
                        # Линия accuracy
                        fig.add_trace(go.Scatter(
                            x=agent_data['timestamp'],
                            y=agent_data['accuracy'],
                            mode='lines+markers',
                            name='Accuracy',
                            line=dict(color='#00D9FF', width=2),
                            marker=dict(size=6)
                        ))
                        
                        # Настройки графика
                        fig.update_layout(
                            title=f"📊 {agent} - Accuracy Over Time",
                            xaxis_title="Time",
                            yaxis_title="Accuracy (%)",
                            template="plotly_dark",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Показываем статистику
                        with st.expander(f"📋 {agent} Statistics"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Avg Accuracy", f"{agent_data['accuracy'].mean():.1f}%")
                            with col2:
                                st.metric("Max Accuracy", f"{agent_data['accuracy'].max():.1f}%")
                            with col3:
                                st.metric("Min Accuracy", f"{agent_data['accuracy'].min():.1f}%")
                else:
                    st.info("📊 Данных по агентам ещё нет.")
            else:
                st.warning("⚠️ Файл performance_summary.csv пуст или повреждён.")
        
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")
            st.info("💡 Попробуй нажать кнопку '🔄 Пересобрать performance_summary.csv'")
    else:
        st.info("📂 Файл performance_summary.csv не найден. Нажми кнопку выше, чтобы создать его.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #8E9297;'>Mode: AI • Model: {} • Powered by Arxora</p>".format(selected_model),
    unsafe_allow_html=True
)
