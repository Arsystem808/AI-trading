# -*- coding: utf-8 -*-
# core/model_fetch.py

import os
import pathlib
import requests
import logging

logging.basicConfig(level=logging.INFO)

# Словарь моделей для загрузки
ASSETS = {
    "alphapulse_AAPL.joblib": os.getenv("MODEL_AAPL_URL"),
    "alphapulse_ETHUSD.joblib": os.getenv("MODEL_ETH_URL"),
}

def ensure_models():
    """
    Скачивает модели из URL, указанных в переменных окружения.
    Модели сохраняются в ARXORA_MODEL_DIR (по умолчанию /tmp/models).
    Пропускает уже скачанные модели.
    """
    # КРИТИЧНО: dest определяется внутри функции
    dest = pathlib.Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
    dest.mkdir(parents=True, exist_ok=True)
    
    for fname, url in ASSETS.items():
        if not url:
            logging.warning(f"Пропускаю {fname}: URL не указан в Secrets")
            continue
        
        path = dest / fname
        
        # Проверяем, существует ли файл и не пустой ли он
        if path.exists() and path.stat().st_size > 0:
            logging.info(f"✓ Модель {fname} уже существует, пропускаю")
            continue
        
        logging.info(f"⬇ Загружаю {fname} из {url[:50]}...")
        
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            path.write_bytes(r.content)
            logging.info(f"✓ Модель {fname} успешно загружена ({len(r.content)} байт)")
        except requests.exceptions.RequestException as e:
            logging.error(f"✗ Ошибка загрузки {fname}: {e}")
            raise
