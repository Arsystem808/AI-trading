import os
import pandas as pd
from datetime import datetime

PERF_DIR = "performance_data"
os.makedirs(PERF_DIR, exist_ok=True)

def log_agent_performance(agent_label: str, ticker: str, date: datetime, daily_return: float):
    """
    Записать доходность агента по определённому тикеру на дату.
    Данные сохраняются в CSV в каталоге performance_data.
    Если запись за дату уже есть — обновляются данные.
    """
    filename = os.path.join(PERF_DIR, f"performance_{agent_label.lower()}_{ticker.upper()}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, parse_dates=["date"])
    else:
        df = pd.DataFrame(columns=["date", "daily_return"])

    date_norm = pd.to_datetime(date).normalize()
    if date_norm in df["date"].values:
        df.loc[df["date"] == date_norm, "daily_return"] = daily_return
    else:
        df = pd.concat([df, pd.DataFrame({"date": [date_norm], "daily_return": [daily_return]})], ignore_index=True)

    df = df.sort_values("date").drop_duplicates("date")
    df.to_csv(filename, index=False)

def get_agent_performance(agent_label: str, ticker: str):
    """
    Получить историю доходности агента по тикеру за последние 90 дней.
    Возвращает DataFrame с columns: date, daily_return, cumulative_return.
    Если данных нет, возвращает None.
    """
    filename = os.path.join(PERF_DIR, f"performance_{agent_label.lower()}_{ticker.upper()}.csv")
    if not os.path.exists(filename):
        return None

    df = pd.read_csv(filename, parse_dates=["date"])
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=90)
    df = df[df["date"] >= cutoff].copy()
    if df.empty:
        return None

    df = df.sort_values("date")
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df
