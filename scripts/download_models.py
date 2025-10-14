#!/usr/bin/env python3
"""
Автоматическое скачивание моделей из GitHub Releases
Использование: python scripts/download_models.py
"""

import requests
import tarfile
import sys
from pathlib import Path

# Настройки
REPO = "Arsystem808/AI-trading"
MODELS_DIR = Path("models")
EXTRACT_DIR = Path(".")

def download_latest_models():
    """Скачать и установить последние модели"""
    
    print("🔍 Поиск последней версии моделей...")
    
    # Получить последний release
    url = f"https://api.github.com/repos/{REPO}/releases/latest"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Ошибка при получении информации о релизе: {e}")
        sys.exit(1)
    
    release = response.json()
    tag = release['tag_name']
    date = release['published_at'][:10]
    
    print(f"✅ Найден релиз: {tag} ({date})")
    
    # Найти архив с моделями
    asset = None
    for a in release['assets']:
        if a['name'].endswith('.tar.gz') and 'sha256' not in a['name']:
            asset = a
            break
    
    if not asset:
        print("❌ Архив с моделями не найден!")
        sys.exit(1)
    
    filename = asset['name']
    download_url = asset['browser_download_url']
    size_mb = asset['size'] / 1024 / 1024
    
    print(f"\n📦 Скачивание: {filename} ({size_mb:.1f} MB)")
    print(f"🔗 URL: {download_url}")
    
    # Скачать архив
    try:
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Показать прогресс
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Прогресс-бар
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)
        
        print(f"\n✅ Скачано: {filename}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при скачивании: {e}")
        sys.exit(1)
    
    # Распаковать архив
    print(f"\n📂 Распаковка архива...")
    
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(EXTRACT_DIR)
        
        print("✅ Архив распакован!")
        
    except Exception as e:
        print(f"❌ Ошибка при распаковке: {e}")
        sys.exit(1)
    
    # Проверить установленные модели
    print(f"\n🔍 Проверка моделей...\n")
    
    if MODELS_DIR.exists():
        models = list(MODELS_DIR.glob('alphapulse_*.joblib'))
        
        if models:
            print(f"✅ Установлено {len(models)} моделей:\n")
            
            for model in sorted(models):
                size_kb = model.stat().st_size / 1024
                print(f"  📦 {model.name} ({size_kb:.1f} KB)")
        else:
            print("⚠️  Модели alphapulse не найдены!")
    else:
        print("❌ Папка models/ не найдена!")
    
    # Удалить архив
    try:
        Path(filename).unlink()
        print(f"\n🗑️  Удален временный файл: {filename}")
    except Exception as e:
        print(f"⚠️  Не удалось удалить {filename}: {e}")
    
    print("\n✅ Готово! Модели установлены.\n")

if __name__ == '__main__':
    download_latest_models()
