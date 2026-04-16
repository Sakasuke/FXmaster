"""Data layer: fetching and caching candles from the broker."""
from fxmaster.data.fetcher import OandaDataFetcher, Granularity
from fxmaster.data.candles import load_candles_csv, save_candles_csv
# ctrader_fetcher は Twisted 依存のため遅延インポート推奨
# from fxmaster.data.ctrader_fetcher import CTraderClient, CTraderCredentials

__all__ = [
    "OandaDataFetcher",
    "Granularity",
    "load_candles_csv",
    "save_candles_csv",
]
