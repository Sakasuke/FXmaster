"""Event-driven-ish backtest engine.

Walks through the LTF candle series bar by bar. On each bar:
  1. Ask the strategy whether to enter (if flat).
  2. If we are in a position, evaluate intrabar whether SL, TP or momentum exit
     triggers. We use the bar's high/low to detect stop-outs and take-profits
     conservatively (SL checked before TP on adverse moves).
  3. Apply costs: spread + optional commission per trade.

This is a simplified model but it mirrors the live runner's decision loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from fxmaster.config import RiskConfig, StrategyConfig
from fxmaster.features.engineer import FeatureBundle
from fxmaster.risk.manager import RiskManager
from fxmaster.strategy.trend_scalp import Direction, Signal, TrendScalpStrategy
from fxmaster.utils.pip_math import pip_size, price_to_pips

logger = logging.getLogger("fxmaster.backtest")


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: Direction
    entry_price: float
    exit_price: float
    units: int
    pnl: float
    pnl_pips: float
    reason_exit: str


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    starting_equity: float = 0.0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if not self.trades:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.starting_equity,
            }
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        equity_values = [v for _, v in self.equity_curve]
        if equity_values:
            running_max = np.maximum.accumulate(equity_values)
            drawdowns = (np.asarray(equity_values) - running_max) / running_max
            max_dd = float(drawdowns.min()) if len(drawdowns) else 0.0
            final_equity = equity_values[-1]
        else:
            max_dd = 0.0
            final_equity = self.starting_equity
        return {
            "trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades),
            "total_pnl": sum(t.pnl for t in self.trades),
            "avg_pnl": sum(t.pnl for t in self.trades) / len(self.trades),
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "max_drawdown": max_dd,
            "final_equity": final_equity,
        }


class BacktestEngine:
    def __init__(
        self,
        strategy: TrendScalpStrategy,
        strategy_cfg: StrategyConfig,
        risk_cfg: RiskConfig,
        instrument: str,
        starting_equity: float = 1_000_000.0,
        spread_pips: float = 0.8,
    ):
        self.strategy = strategy
        self.strategy_cfg = strategy_cfg
        self.instrument = instrument
        self.risk_manager = RiskManager(risk_cfg, starting_equity)
        self.spread_price = spread_pips * pip_size(instrument)
        self.starting_equity = starting_equity

    def run(
        self,
        candles: pd.DataFrame,
        bundle: FeatureBundle,
    ) -> BacktestResult:
        """Run the backtest over the aligned candles/feature bundle."""
        if len(candles) != len(bundle.features):
            raise ValueError(
                f"Candles ({len(candles)}) and features ({len(bundle.features)}) length mismatch. "
                "They must be produced from the same LTF series."
            )

        result = BacktestResult(starting_equity=self.starting_equity)
        position_direction: Direction = Direction.FLAT
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        partial_taken = False
        units = 0
        entry_time: datetime | None = None

        prev_hist = 0.0

        features = bundle.features.reset_index(drop=True)
        for i in range(len(candles)):
            bar = candles.iloc[i]
            bar_time = pd.to_datetime(bar["time"], utc=True).to_pydatetime()
            row = features.iloc[i]
            current_hist = float(row.get("ltf_macd_hist", 0.0) or 0.0)

            # --- manage open position intrabar -----------------------------
            if position_direction != Direction.FLAT:
                exit_price: float | None = None
                reason_exit = ""
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])

                if position_direction == Direction.LONG:
                    if bar_low <= stop_loss:
                        exit_price = stop_loss
                        reason_exit = "stop-loss"
                    elif bar_high >= take_profit:
                        exit_price = take_profit
                        reason_exit = "take-profit"
                else:
                    if bar_high >= stop_loss:
                        exit_price = stop_loss
                        reason_exit = "stop-loss"
                    elif bar_low <= take_profit:
                        exit_price = take_profit
                        reason_exit = "take-profit"

                # Momentum-based early exit (only if SL/TP not triggered)
                if exit_price is None and self.strategy.should_momentum_exit(
                    position_direction, prev_hist, current_hist
                ):
                    exit_price = float(bar["close"])
                    reason_exit = "momentum-exit"

                # Trailing stop update once partial target is reached
                atr_value = float(row.get("ltf_atr", 0.0) or 0.0)
                if exit_price is None and atr_value > 0:
                    if position_direction == Direction.LONG:
                        r_distance = entry_price - stop_loss
                        if not partial_taken and r_distance > 0:
                            r_now = (float(bar["close"]) - entry_price) / r_distance
                            if r_now >= self.strategy_cfg.partial_take_profit_at_r:
                                partial_taken = True
                                stop_loss = entry_price  # move to breakeven
                        if partial_taken:
                            new_stop = float(bar["close"]) - atr_value * self.strategy_cfg.trailing_atr_mult
                            stop_loss = max(stop_loss, new_stop)
                    else:
                        r_distance = stop_loss - entry_price
                        if not partial_taken and r_distance > 0:
                            r_now = (entry_price - float(bar["close"])) / r_distance
                            if r_now >= self.strategy_cfg.partial_take_profit_at_r:
                                partial_taken = True
                                stop_loss = entry_price
                        if partial_taken:
                            new_stop = float(bar["close"]) + atr_value * self.strategy_cfg.trailing_atr_mult
                            stop_loss = min(stop_loss, new_stop)

                if exit_price is not None and entry_time is not None:
                    pnl_price = (
                        (exit_price - entry_price) * units
                        if position_direction == Direction.LONG
                        else (entry_price - exit_price) * units
                    )
                    # Subtract round-trip spread cost
                    pnl_price -= self.spread_price * abs(units)
                    pnl_pips = price_to_pips(
                        (exit_price - entry_price) * int(position_direction),
                        self.instrument,
                    )
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=bar_time,
                        direction=position_direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        units=units,
                        pnl=pnl_price,
                        pnl_pips=pnl_pips,
                        reason_exit=reason_exit,
                    )
                    result.trades.append(trade)
                    self.risk_manager.register_close(pnl_price, now=bar_time)
                    result.equity_curve.append((bar_time, self.risk_manager.state.equity))
                    position_direction = Direction.FLAT
                    partial_taken = False
                    entry_time = None
                    units = 0

            # --- try to open a new position --------------------------------
            if position_direction == Direction.FLAT:
                can_open, _ = self.risk_manager.can_open_new_position(now=bar_time)
                if can_open:
                    price = float(bar["close"])
                    signal: Signal = self.strategy.evaluate_row(row, price)
                    if signal.is_actionable:
                        trade_units = self.risk_manager.position_size_units(
                            instrument=self.instrument,
                            entry_price=signal.entry_price,
                            stop_price=signal.stop_loss,
                        )
                        if trade_units > 0:
                            position_direction = signal.direction
                            entry_price = signal.entry_price
                            stop_loss = signal.stop_loss
                            take_profit = signal.take_profit
                            units = trade_units if signal.direction == Direction.LONG else -trade_units
                            entry_time = bar_time
                            partial_taken = False
                            self.risk_manager.register_open(now=bar_time)

            prev_hist = current_hist

        # Close any dangling position at the last bar's close
        if position_direction != Direction.FLAT and entry_time is not None:
            last_bar = candles.iloc[-1]
            last_time = pd.to_datetime(last_bar["time"], utc=True).to_pydatetime()
            exit_price = float(last_bar["close"])
            pnl_price = (
                (exit_price - entry_price) * units
                if position_direction == Direction.LONG
                else (entry_price - exit_price) * units
            )
            pnl_price -= self.spread_price * abs(units)
            pnl_pips = price_to_pips(
                (exit_price - entry_price) * int(position_direction),
                self.instrument,
            )
            result.trades.append(
                Trade(
                    entry_time=entry_time,
                    exit_time=last_time,
                    direction=position_direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    units=units,
                    pnl=pnl_price,
                    pnl_pips=pnl_pips,
                    reason_exit="end-of-series",
                )
            )
            self.risk_manager.register_close(pnl_price, now=last_time)
            result.equity_curve.append((last_time, self.risk_manager.state.equity))

        if not result.equity_curve:
            result.equity_curve.append(
                (pd.to_datetime(candles.iloc[-1]["time"], utc=True).to_pydatetime(), self.starting_equity)
            )
        return result


def iter_trades(result: BacktestResult) -> Iterable[Trade]:
    yield from result.trades
