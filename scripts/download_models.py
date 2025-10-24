#!/usr/bin/env python3
"""
Автоматическое скачивание моделей из GitHub Releases

Скачивает последнюю версию моделей из GitHub Releases,
распаковывает архив и проверяет целостность файлов.

Usage:
    python scripts/download_models.py [--tag TAG] [--force]
    
    --tag TAG: Скачать конкретную версию (по умолчанию - latest)
    --force: Перезаписать существующие модели
"""

import os
import argparse
import hashlib
import requests
import tarfile
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Настройки
REPO = "Arsystem808/AI-trading"
MODELS_DIR = Path("models")
CONFIG_DIR = Path("config")
EXTRACT_DIR = Path(".")

def get_release_info(tag: Optional[str] = None) -> Dict[str, Any]:
    """
    Получить информацию о релизе.
    
    Args:
        tag: Тег версии (None = latest)
        
    Returns:
        dict: Данные релиза
    """
    if tag:
        url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    else:
        url = f"https://api.github.com/repos/{REPO}/releases/latest"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"❌ Релиз не найден: {tag or 'latest'}")
        else:
            print(f"❌ HTTP ошибка: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка при получении информации о релизе: {e}")
        sys.exit(1)


def verify_checksum(file_path: Path, expected_hash: Optional[str] = None) -> bool:
    """
    Проверить SHA256 чексумму файла.
    
    Args:
        file_path: Путь к файлу
        expected_hash: Ожидаемый хеш (опционально)
        
    Returns:
        bool: True если проверка прошла
    """
    if not expected_hash:
        return True
    
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    actual_hash = sha256_hash.hexdigest()
    
    if actual_hash.lower() != expected_hash.lower():
        print(f"⚠️  Checksum mismatch!")
        print(f"   Expected: {expected_hash}")
        print(f"   Got:      {actual_hash}")
        return False
    
    print("✅ Checksum verified!")
    return True


def download_file(url: str, filename: str) -> bool:
    """
    Скачать файл с прогресс-баром.
    
    Args:
        url: URL для скачивания
        filename: Имя сохраняемого файла
        
    Returns:
        bool: True если успешно
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
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
        
        print()  # Новая строка после прогресса
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при скачивании: {e}")
        return False


def extract_archive(filename: str, extract_to: Path = EXTRACT_DIR) -> bool:
    """
    Распаковать tar.gz архив.
    
    Args:
        filename: Имя архива
        extract_to: Куда распаковать
        
    Returns:
        bool: True если успешно
    """
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            # Безопасная распаковка (проверка путей)
            members = tar.getmembers()
            
            for member in members:
                # Проверка на path traversal
                if member.name.startswith(('/', '..')):
                    print(f"⚠️  Suspicious path in archive: {member.name}")
                    continue
                
                tar.extract(member, extract_to)
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при распаковке: {e}")
        return False


def verify_models(force: bool = False) -> None:
    """
    Проверить установленные модели.
    
    Args:
        force: Игнорировать существующие файлы
    """
    print(f"\n🔍 Проверка моделей...\n")
    
    if not MODELS_DIR.exists():
        print("❌ Папка models/ не найдена!")
        return
    
    # Поиск моделей
    model_patterns = [
        '*.joblib',
        '*.pkl',
        '**/*.joblib',
        '**/*.pkl'
    ]
    
    all_models = []
    for pattern in model_patterns:
        all_models.extend(MODELS_DIR.glob(pattern))
    
    # Исключить scalers
    models = [
        f for f in all_models 
        if 'scaler' not in f.name.lower()
    ]
    
    if not models:
        print("⚠️  Модели не найдены!")
        return
    
    print(f"✅ Установлено {len(models)} моделей:\n")
    
    # Группировка по агенту
    by_agent = {}
    for model in sorted(models):
        # Извлечь имя агента из пути
        parts = model.parts
        agent = parts[-2] if len(parts) > 1 else "root"
        
        if agent not in by_agent:
            by_agent[agent] = []
        
        by_agent[agent].append(model)
    
    # Вывод по агентам
    for agent, agent_models in sorted(by_agent.items()):
        print(f"📁 {agent}:")
        
        for model in agent_models:
            size_kb = model.stat().st_size / 1024
            print(f"  📦 {model.name} ({size_kb:.1f} KB)")
        
        print()
    
    # Проверить конфиги
    if CONFIG_DIR.exists():
        configs = list(CONFIG_DIR.glob('*.json'))
        print(f"📋 Найдено {len(configs)} конфигураций")


def cleanup(filename: str) -> None:
    """
    Удалить временные файлы.
    
    Args:
        filename: Файл для удаления
    """
    try:
        Path(filename).unlink()
        print(f"🗑️  Удален временный файл: {filename}")
    except Exception as e:
        print(f"⚠️  Не удалось удалить {filename}: {e}")


def download_latest_models(tag: Optional[str] = None, force: bool = False) -> None:
    """
    Скачать и установить модели из GitHub Release.
    
    Args:
        tag: Конкретная версия (None = latest)
        force: Перезаписать существующие модели
    """
    print("🔍 Поиск релиза с моделями...")
    
    # Получить релиз
    release = get_release_info(tag)
    
    release_tag = release['tag_name']
    release_date = release['published_at'][:10]
    release_name = release.get('name', release_tag)
    
    print(f"✅ Найден релиз: {release_name} ({release_tag}, {release_date})")
    
    # Найти архив с моделями
    archive_asset = None
    checksum_asset = None
    
    for asset in release['assets']:
        name = asset['name']
        
        if name.endswith('.tar.gz') and 'sha256' not in name:
            archive_asset = asset
        elif name.endswith('.sha256') or 'sha256' in name:
            checksum_asset = asset
    
    if not archive_asset:
        print("❌ Архив с моделями не найден в релизе!")
        sys.exit(1)
    
    filename = archive_asset['name']
    download_url = archive_asset['browser_download_url']
    size_mb = archive_asset['size'] / 1024 / 1024
    
    print(f"\n📦 Скачивание: {filename} ({size_mb:.1f} MB)")
    print(f"🔗 URL: {download_url}")
    
    # Проверка существующих файлов
    if MODELS_DIR.exists() and not force:
        existing = list(MODELS_DIR.glob('*.joblib')) + list(MODELS_DIR.glob('*.pkl'))
        
        if existing:
            # Проверка CI окружения
            # GitHub Actions устанавливает GITHUB_ACTIONS=true
            # Большинство CI систем устанавливают CI=true
            is_ci = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"
            
            if is_ci:
                # В CI автоматически продолжаем
                print(f"\n⚠️  Найдено {len(existing)} существующих моделей")
                print("   🤖 CI environment detected - продолжаем автоматически")
            else:
                # В локальной среде спрашиваем пользователя
                print(f"\n⚠️  Найдено {len(existing)} существующих моделей")
                print("   Используйте --force для перезаписи")
                
                response = input("Продолжить? (y/N): ")
                if response.lower() != 'y':
                    print("❌ Отменено")
                    sys.exit(0)
    
    # Скачать архив
    if not download_file(download_url, filename):
        sys.exit(1)
    
    print(f"✅ Скачано: {filename}")
    
    # Скачать и проверить checksum (если есть)
    expected_hash = None
    
    if checksum_asset:
        checksum_url = checksum_asset['browser_download_url']
        checksum_file = checksum_asset['name']
        
        print(f"\n📥 Скачивание checksums...")
        
        if download_file(checksum_url, checksum_file):
            try:
                with open(checksum_file, 'r') as f:
                    # Формат: <hash> <filename>
                    line = f.read().strip().split()
                    expected_hash = line[0]
                
                Path(checksum_file).unlink()
            except Exception as e:
                print(f"⚠️  Не удалось прочитать checksum: {e}")
    
    # Проверить checksum
    if expected_hash:
        print(f"\n🔐 Проверка целостности...")
        
        if not verify_checksum(Path(filename), expected_hash):
            print("❌ Checksum не совпадает! Файл может быть поврежден.")
            cleanup(filename)
            sys.exit(1)
    
    # Распаковать архив
    print(f"\n📂 Распаковка архива...")
    
    if not extract_archive(filename):
        cleanup(filename)
        sys.exit(1)
    
    print("✅ Архив распакован!")
    
    # Проверить модели
    verify_models(force)
    
    # Удалить архив
    cleanup(filename)
    
    print("\n✅ Готово! Модели установлены.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download models from GitHub Releases",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--tag',
        type=str,
        help='Specific release tag (default: latest)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing models'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available releases'
    )
    
    args = parser.parse_args()
    
    # Список релизов
    if args.list:
        print("🔍 Получение списка релизов...")
        
        try:
            url = f"https://api.github.com/repos/{REPO}/releases"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            releases = response.json()
            
            print(f"\n📋 Доступно релизов: {len(releases)}\n")
            
            for release in releases[:10]:  # Показать 10 последних
                tag = release['tag_name']
                date = release['published_at'][:10]
                name = release.get('name', tag)
                
                print(f"  • {name} ({tag}) - {date}")
            
            print()
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            sys.exit(1)
        
        return
    
    # Скачать модели
    download_latest_models(tag=args.tag, force=args.force)


if __name__ == '__main__':
    main()
