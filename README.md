[README_UPDATE.txt](https://github.com/user-attachments/files/22123685/README_UPDATE.txt)

ARXORA — PATCH PACK (не ломает существующий код)

1) Скопируйте файлы в свой проект:
   - assets/arxora_logo.png  -> в вашу папку ассетов (например, ./assets/)
   - branding.py             -> в корень проекта (рядом с app.py)
   - utils/tickers.py        -> в папку ./utils/ (создайте если нет)

2) В app.py:
   2.1) Установите заголовок и favicon и подключите шапку Arxora:

   from branding import render_header
   st.set_page_config(page_title="Arxora", page_icon="assets/arxora_logo.png", layout="wide")
   render_header("assets/arxora_logo.png", "trade smarter.")

   2.2) В месте ввода тикера добавьте placeholder:
   ticker = st.text_input("Тикер", value="AAPL", placeholder="Примеры: AAPL, TSLA, X:BTCUSD, BINANCE:ETHUSDT")

3) В вашем загрузчике данных нормализуйте тикер:
   from utils.tickers import normalize_to_yf
   yf_symbol = normalize_to_yf(ticker)
   # дальше используйте yf_symbol в yfinance (или аналог)
# temp check пятница,  3 октября 2025 г. 15:26:41 (MSK)
