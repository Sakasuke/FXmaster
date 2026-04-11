"""Trend-following scalp strategy inspired by Mizushima-style multi-timeframe analysis.

Entry logic (per LTF bar):
  * HTF trend direction (EMA fast vs slow) gives the bias.
  * HTF ADX >= min_adx filters out range markets.
  * LTF RSI pullback confirms the mean-reversion-to-trend entry.
  * AI confidence (probability of a meaningful move in the next N bars) must
    exceed the configured threshold.
  * Current spread must not exceed the configured cap.

Exit logic is split across:
  * Hard stop: -1R (computed from entry price and ATR-based stop distance)
  * Partial profit: take ``partial_take_profit_ratio`` of position off at +1.5R
  * Trailing: once partial is taken, trail the remaining position using
    ``trailing_atr_mult * ATR``.
  * Momentum exit: close early when MACD hist starts shrinking against us.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxmaster.ai.model import TrendModel
from fxmaster.config import StrategyConfig

logger = logging.getLogger("fxmaster.strategy")


class Direction(enum.IntEnum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Signal:
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    reason: str

    @property
    def is_actionable(self) -> bool:
        return self.direction != Direction.FLAT


class TrendScalpStrategy:
    def __init__(self, cfg: StrategyConfig, model: TrendModel | None = None):
        self.cfg = cfg
        self.model = model

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def evaluate_row(self, features_row: pd.Series, price: float) -> Signal:
        """Evaluate a single LTF bar and return a signal object.

        ``features_row`` is expected to contain every column produced by
        ``fxmaster.features.engineer.build_features``.
        """
        atr_value = float(features_row.get("ltf_atr", np.nan))
        if not np.isfinite(atr_value) or atr_value <= 0:
            return self._flat("ATR not ready")

        htf_strong = bool(features_row.get("htf_strong_trend", 0))
        if not htf_strong:
            return self._flat("HTF trend too weak (ADX filter)")

        uptrend = bool(features_row.get("htf_uptrend", 0))
        pullback_long = bool(features_row.get("ltf_pullback_long", 0))
        pullback_short = bool(features_row.get("ltf_pullback_short", 0))

        if uptrend and pullback_long:
            direction = Direction.LONG
        elif (not uptrend) and pullback_short:
            direction = Direction.SHORT
        else:
            return self._flat("No pullback trigger")

        confidence = self._ai_confidence(features_row)
        if confidence < self.cfg.ai_confidence_min:
            return self._flat(f"AI confidence {confidence:.2f} below threshold")

        sl_distance = atr_value * self.cfg.atr_sl_mult
        tp_distance = atr_value * self.cfg.atr_tp_mult
        if direction == Direction.LONG:
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:
            stop_loss = price + sl_distance
            take_profit = price - tp_distance

        return Signal(
            direction=direction,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr_value,
            reason="HTF trend + LTF pullback + AI confirmed",
        )

    def _ai_confidence(self, features_row: pd.Series) -> float:
        if self.model is None:
            # No AI loaded: fall back to a purely rule-based signal with a
            # conservative confidence that still lets the pipeline run in tests.
            return max(self.cfg.ai_confidence_min, 0.55)
        row_df = features_row.to_frame().T
        prob = float(self.model.predict_proba(row_df)[0])
        return prob

    def _flat(self, reason: str) -> Signal:
        return Signal(
            direction=Direction.FLAT,
            confidence=0.0,
            entry_price=float("nan"),
            stop_loss=float("nan"),
            take_profit=float("nan"),
            atr=float("nan"),
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Exit helpers (used by backtest and live runner)
    # ------------------------------------------------------------------

    def should_momentum_exit(
        self,
        direction: Direction,
        prev_hist: float,
        current_hist: float,
    ) -> bool:
        """Return True if MACD histogram momentum is weakening against us."""
        if direction == Direction.LONG:
            return prev_hist > 0 and current_hist < prev_hist * 0.5
        if direction == Direction.SHORT:
            return prev_hist < 0 and current_hist > prev_hist * 0.5
        return False

    def trailing_stop(self, direction: Direction, price: float, atr_value: float) -> float:
        offset = atr_value * self.cfg.trailing_atr_mult
        return price - offset if direction == Direction.LONG else price + offset
