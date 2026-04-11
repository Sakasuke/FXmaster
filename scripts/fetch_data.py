"""Download historical candles from OANDA and cache them as CSV.

Usage:
    python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 365
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxmaster.config import load_config
from fxmaster.data import OandaDataFetcher, save_candles_csv
from fxmaster.data.fetcher import OandaCredentials
from fxmaster.utils.logger import get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch candles from OANDA and cache them locally.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--instrument", default=None, help="Override config instrument")
    parser.add_argument("--granularity", default=None, help="e.g. M5, M15, H1")
    parser.add_argument("--days", type=int, default=365, help="How many days of history to fetch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("fxmaster.fetch", cfg.logging.level, cfg.logging.file)

    instrument = args.instrument or cfg.trading.instrument
    granularity = args.granularity or cfg.trading.ltf_granularity
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    creds = OandaCredentials(
        account_id=cfg.broker.account_id,
        access_token=cfg.broker.access_token,
        environment=cfg.broker.environment,
    )
    fetcher = OandaDataFetcher(creds)

    logger.info("Fetching %s %s from %s to %s", instrument, granularity, start, end)
    df = fetcher.fetch_candles(instrument, granularity, start, end)
    logger.info("Received %d candles", len(df))

    out_path = Path(cfg.data.cache_dir) / f"{instrument}_{granularity}.csv"
    save_candles_csv(df, out_path)
    logger.info("Saved to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
