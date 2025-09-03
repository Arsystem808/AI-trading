[README.md](https://github.com/user-attachments/files/22102641/README.md)
# Arxora TradeAI

Эта версия использует **Polygon.io** как единственный источник данных.

## Быстрый старт
1. Создайте файл `.env` в корне:
```
POLYGON_API_KEY=ooSjpJAULw4VXxsY28ck7DQST7i13kcG
```
2. Установите зависимости и запустите:
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Что учтено
- Полностью убран Yahoo/yfinance.
- Нет текста «Говорит, как опытный трейдер…».
- UI стилизован под карточки: крупная цена, **Buy LONG / Sell SHORT / WAIT**, % confidence, Entry/SL/TP1/TP2/TP3 с вероятностями.
- Нет упоминаний количества баров и выбора окон.
- Альтернативный план формулируется без слова «Альт».
