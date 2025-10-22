# core/logger.py
import logging
import os

LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

# если logging уже был сконфигурирован, force=True перезапишет конфиг (py3.8+)
logging.basicConfig(level=getattr(logging, LEVEL, logging.INFO), format=FMT, force=True)

logger = logging.getLogger("arxora")
logger.setLevel(getattr(logging, LEVEL, logging.INFO))
