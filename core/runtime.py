from services.live_quotes import LiveQuoteService
quotes = LiveQuoteService(ttl_sec=2.5)  # без use_ws
