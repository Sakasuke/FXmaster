"""Train the LightGBM trend continuation model from cached candles.

Usage:
    python scripts/train_model.py --instrument USD_JPY
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxmaster.ai.trainer import train_model
from fxmaster.config import load_config
from fxmaster.data import load_candles_csv
from fxmaster.features.engineer import build_features, drop_na_rows
from fxmaster.utils.logger import get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the trend continuation model.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--instrument", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("fxmaster.train", cfg.logging.level, cfg.logging.file)

    instrument = args.instrument or cfg.trading.instrument
    ltf_path = Path(cfg.data.cache_dir) / f"{instrument}_{cfg.trading.ltf_granularity}.csv"
    htf_path = Path(cfg.data.cache_dir) / f"{instrument}_{cfg.trading.htf_granularity}.csv"

    if not ltf_path.exists() or not htf_path.exists():
        logger.error(
            "Missing candle cache. Run fetch_data.py for both %s and %s first.",
            cfg.trading.ltf_granularity,
            cfg.trading.htf_granularity,
        )
        return 1

    ltf = load_candles_csv(ltf_path)
    htf = load_candles_csv(htf_path)
    logger.info("Loaded %d LTF and %d HTF candles", len(ltf), len(htf))

    bundle = build_features(ltf, htf, instrument, cfg.strategy, cfg.model)
    bundle = drop_na_rows(bundle)
    logger.info("Feature matrix after NA drop: %s", bundle.features.shape)

    model = train_model(bundle, cfg.model, instrument)
    saved = model.save(cfg.model.model_dir, name=f"{instrument}_trend")
    logger.info("Saved model to %s", saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
