# core/runtime.py
from services.live_quotes import LiveQuoteService
quotes = LiveQuoteService(use_ws=True, ttl_sec=2.5)
