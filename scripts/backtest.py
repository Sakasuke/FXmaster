"""Run a backtest on cached candle data.

Usage:
    python scripts/backtest.py --instrument USD_JPY --from 2024-01-01 --to 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxmaster.ai.model import TrendModel
from fxmaster.backtest import BacktestEngine
from fxmaster.config import load_config
from fxmaster.data import load_candles_csv
from fxmaster.features.engineer import build_features
from fxmaster.strategy import TrendScalpStrategy
from fxmaster.utils.logger import get_logger


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the trend-scalp strategy.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--from", dest="date_from", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--equity", type=float, default=1_000_000.0, help="Starting equity (JPY)")
    parser.add_argument("--spread", type=float, default=0.8, help="Assumed spread in pips")
    parser.add_argument("--no-model", action="store_true", help="Run without AI model (rules only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("fxmaster.backtest", cfg.logging.level, cfg.logging.file)

    instrument = args.instrument or cfg.trading.instrument
    ltf_path = Path(cfg.data.cache_dir) / f"{instrument}_{cfg.trading.ltf_granularity}.csv"
    htf_path = Path(cfg.data.cache_dir) / f"{instrument}_{cfg.trading.htf_granularity}.csv"

    ltf = load_candles_csv(ltf_path)
    htf = load_candles_csv(htf_path)

    date_from = _parse_date(args.date_from)
    date_to = _parse_date(args.date_to)
    if date_from is not None:
        ltf = ltf[ltf["time"] >= pd.Timestamp(date_from)].reset_index(drop=True)
    if date_to is not None:
        ltf = ltf[ltf["time"] <= pd.Timestamp(date_to)].reset_index(drop=True)

    bundle = build_features(ltf, htf, instrument, cfg.strategy, cfg.model)
    # Drop rows with NaN features (burn-in), keep alignment with candles
    mask = bundle.features.notna().all(axis=1)
    ltf = ltf.loc[mask].reset_index(drop=True)
    bundle.features.reset_index(drop=True, inplace=True)
    bundle.features = bundle.features.loc[mask].reset_index(drop=True)
    bundle.label = bundle.label.loc[mask].reset_index(drop=True)
    bundle.close = bundle.close.loc[mask].reset_index(drop=True)

    model = None
    if not args.no_model:
        try:
            model = TrendModel.load(cfg.model.model_dir, name=f"{instrument}_trend")
            logger.info("Loaded AI model trained on %s", model.trained_on)
        except FileNotFoundError:
            logger.warning("No AI model found; running with rule-based fallback.")

    strategy = TrendScalpStrategy(cfg.strategy, model=model)
    engine = BacktestEngine(
        strategy=strategy,
        strategy_cfg=cfg.strategy,
        risk_cfg=cfg.risk,
        instrument=instrument,
        starting_equity=args.equity,
        spread_pips=args.spread,
    )

    result = engine.run(ltf, bundle)
    summary = result.summary()
    logger.info("Backtest summary:")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
