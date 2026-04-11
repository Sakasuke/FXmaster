"""Live/paper trading runner.

Usage:
    python scripts/run_live.py --instrument USD_JPY

Operational notes:
  * Run this on a VPS with persistent network connectivity.
  * ALWAYS start on OANDA practice (demo) environment before live.
  * The runner polls price + features every LTF bar length; for M5 that is 5
    minutes. This is intentionally slow and boring — it avoids overtrading.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxmaster.ai.model import TrendModel
from fxmaster.broker import OandaBroker
from fxmaster.config import load_config
from fxmaster.data import OandaDataFetcher
from fxmaster.data.fetcher import OandaCredentials, granularity_to_seconds
from fxmaster.features.engineer import build_features
from fxmaster.risk.manager import RiskManager
from fxmaster.strategy import Direction, TrendScalpStrategy
from fxmaster.utils.logger import get_logger
from fxmaster.utils.pip_math import pip_size, price_to_pips


def _trading_hours_ok(now: datetime, start_str: str, end_str: str) -> bool:
    def _parse(hm: str) -> tuple[int, int]:
        h, m = hm.split(":")
        return int(h), int(m)

    sh, sm = _parse(start_str)
    eh, em = _parse(end_str)
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now <= end


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the live/paper trading loop.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--env", default=None, choices=["practice", "live"],
                        help="Override broker environment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate signals but do not place orders")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("fxmaster.live", cfg.logging.level, cfg.logging.file)

    if args.env:
        cfg.broker.environment = args.env
    instrument = args.instrument or cfg.trading.instrument

    creds = OandaCredentials(
        account_id=cfg.broker.account_id,
        access_token=cfg.broker.access_token,
        environment=cfg.broker.environment,
    )
    fetcher = OandaDataFetcher(creds)
    broker = OandaBroker(creds)

    account = broker.account_info()
    logger.info(
        "Connected to OANDA %s. Account %s balance %.2f %s",
        cfg.broker.environment,
        account.account_id,
        account.balance,
        account.currency,
    )

    try:
        model = TrendModel.load(cfg.model.model_dir, name=f"{instrument}_trend")
        logger.info("AI model loaded (trained %s)", model.trained_on)
    except FileNotFoundError:
        model = None
        logger.warning("No AI model found; using rule-based fallback. Train one before going live.")

    strategy = TrendScalpStrategy(cfg.strategy, model=model)
    risk = RiskManager(cfg.risk, initial_equity=account.equity)

    bar_seconds = granularity_to_seconds(cfg.trading.ltf_granularity)
    # Look back enough bars to warm up ALL indicators.
    lookback_bars = max(cfg.strategy.ema_slow * 3, 200)
    htf_lookback_bars = max(cfg.strategy.ema_slow * 3, 200)
    htf_seconds = granularity_to_seconds(cfg.trading.htf_granularity)

    logger.info("Entering main loop (Ctrl-C to stop)")
    try:
        while True:
            now = datetime.now(timezone.utc)
            if not _trading_hours_ok(
                now, cfg.trading.trading_hours_utc.start, cfg.trading.trading_hours_utc.end
            ):
                logger.debug("Outside trading hours; sleeping")
                time.sleep(60)
                continue

            ltf_start = now - timedelta(seconds=bar_seconds * lookback_bars)
            htf_start = now - timedelta(seconds=htf_seconds * htf_lookback_bars)
            ltf = fetcher.fetch_candles(instrument, cfg.trading.ltf_granularity, ltf_start, now)
            htf = fetcher.fetch_candles(instrument, cfg.trading.htf_granularity, htf_start, now)

            if len(ltf) < 50 or len(htf) < 50:
                logger.warning("Not enough candles yet (ltf=%d htf=%d)", len(ltf), len(htf))
                time.sleep(bar_seconds)
                continue

            bundle = build_features(ltf, htf, instrument, cfg.strategy, None)
            row = bundle.features.iloc[-1]
            price = float(ltf["close"].iloc[-1])

            # Spread filter
            bid, ask = broker.get_price(instrument)
            spread_pips = price_to_pips(ask - bid, instrument)
            if spread_pips > cfg.trading.max_spread_pips:
                logger.info("Spread %.2f pips > limit; skipping", spread_pips)
                time.sleep(bar_seconds)
                continue

            open_pos = broker.get_open_position(instrument)
            risk.state.open_positions = 1 if open_pos else 0

            signal = strategy.evaluate_row(row, price)
            logger.info(
                "bar=%s dir=%s conf=%.2f reason=%s price=%.5f",
                ltf["time"].iloc[-1],
                signal.direction.name,
                signal.confidence,
                signal.reason,
                price,
            )

            can_open, reason = risk.can_open_new_position(now=now)
            if (
                signal.is_actionable
                and open_pos is None
                and can_open
            ):
                units = risk.position_size_units(
                    instrument=instrument,
                    entry_price=signal.entry_price,
                    stop_price=signal.stop_loss,
                )
                if units <= 0:
                    logger.info("Sized 0 units; skipping")
                else:
                    signed_units = units if signal.direction == Direction.LONG else -units
                    if args.dry_run:
                        logger.info("[DRY-RUN] Would place order units=%d SL=%.5f TP=%.5f",
                                    signed_units, signal.stop_loss, signal.take_profit)
                    else:
                        order = broker.place_market_order(
                            instrument=instrument,
                            units=signed_units,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                        )
                        risk.register_open(now=now)
                        logger.info(
                            "Opened position id=%s units=%d at %.5f",
                            order.order_id,
                            order.units,
                            order.price,
                        )
            elif not can_open:
                logger.debug("Risk gate closed: %s", reason)

            time.sleep(bar_seconds)
    except KeyboardInterrupt:
        logger.info("Interrupted by user; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
