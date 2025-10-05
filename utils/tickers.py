# utils/tickers.py — normalizer for crypto tickers typed like X:BTCUSD or BINANCE:ETHUSDT
def normalize_to_yf(symbol: str) -> str:
    """
    Examples:
      "X:BTCUSD"         -> "BTC-USD"
      "BINANCE:ETHUSDT"  -> "ETH-USD"
      "CRYPTO:SOLUSD"    -> "SOL-USD"
      "BTCUSDT"          -> "BTC-USD"
      "ETHUSD"           -> "ETH-USD"
      "AAPL"             -> "AAPL"
    """
    s = (symbol or "").strip().upper().replace(" ", "")

    for pref in (
        "X:",
        "CRYPTO:",
        "BINANCE:",
        "COINBASE:",
        "KRAKEN:",
        "BYBIT:",
        "HUOBI:",
        "OKX:",
    ):
        if s.startswith(pref):
            s = s[len(pref) :]

    if s.endswith("USDT"):
        s = s[:-4] + "USD"

    if len(s) >= 6 and s.endswith("USD"):
        return s[:-3] + "-USD"

    return s
