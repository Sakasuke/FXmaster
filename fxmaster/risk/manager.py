"""Position sizing, daily loss caps, and consecutive loss cooldowns.

Rules:
  * Per-trade risk is a fixed percentage of current account equity.
  * Position size is computed so that a stop-out loses exactly that amount.
  * When the daily loss threshold is breached, no new trades until next day.
  * After ``max_consecutive_losses`` consecutive losing trades, trading is
    paused for ``cooldown_minutes_after_max_losses`` minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from fxmaster.config import RiskConfig
from fxmaster.utils.pip_math import pip_size


@dataclass
class RiskState:
    equity: float
    day: str = ""
    day_start_equity: float = 0.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    paused_until: datetime | None = None
    open_positions: int = 0
    trade_log: list[dict] = field(default_factory=list)


class RiskManager:
    def __init__(self, cfg: RiskConfig, initial_equity: float):
        self.cfg = cfg
        self.state = RiskState(equity=initial_equity, day_start_equity=initial_equity)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def position_size_units(
        self,
        instrument: str,
        entry_price: float,
        stop_price: float,
        account_currency_rate: float = 1.0,
    ) -> int:
        """Return the number of units to trade.

        ``account_currency_rate`` is the rate used to convert the quote-currency
        loss into the account currency. For a JPY account trading USD/JPY this
        is simply 1 (the quote currency IS the account currency). For
        EUR/USD with a JPY account it would be the USD/JPY rate.
        """
        risk_amount = self.state.equity * self.cfg.risk_per_trade_pct
        price_distance = abs(entry_price - stop_price)
        if price_distance <= 0:
            return 0
        # Loss per 1 unit = price_distance (in quote currency)
        loss_per_unit = price_distance * account_currency_rate
        if loss_per_unit <= 0:
            return 0
        units = int(risk_amount // loss_per_unit)
        # Round down to nearest lot of 1000 to match OANDA's minimum
        units = (units // 1000) * 1000
        return max(units, 0)

    # ------------------------------------------------------------------
    # Gate: can we trade right now?
    # ------------------------------------------------------------------

    def can_open_new_position(self, now: datetime | None = None) -> tuple[bool, str]:
        now = now or datetime.now(timezone.utc)
        self._roll_day_if_needed(now)

        if self.state.open_positions >= self.cfg.max_concurrent_positions:
            return False, "max concurrent positions reached"
        if self.state.paused_until and now < self.state.paused_until:
            remaining = self.state.paused_until - now
            return False, f"cooldown for {int(remaining.total_seconds()/60)}m"
        max_daily_loss = self.state.day_start_equity * self.cfg.max_daily_loss_pct
        if self.state.daily_pnl <= -max_daily_loss:
            return False, "daily loss cap reached"
        return True, "ok"

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def register_open(self, now: datetime | None = None) -> None:
        self._roll_day_if_needed(now or datetime.now(timezone.utc))
        self.state.open_positions += 1

    def register_close(self, pnl: float, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        self._roll_day_if_needed(now)
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.equity += pnl
        self.state.daily_pnl += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.state.paused_until = now + timedelta(
                    minutes=self.cfg.cooldown_minutes_after_max_losses
                )
        else:
            self.state.consecutive_losses = 0
        self.state.trade_log.append(
            {
                "time": now.isoformat(),
                "pnl": pnl,
                "equity_after": self.state.equity,
            }
        )

    def _roll_day_if_needed(self, now: datetime) -> None:
        today = now.strftime("%Y-%m-%d")
        if self.state.day != today:
            self.state.day = today
            self.state.day_start_equity = self.state.equity
            self.state.daily_pnl = 0.0
            # Clear pause if it lapsed before the new day.
            if self.state.paused_until and now >= self.state.paused_until:
                self.state.paused_until = None


__all__ = ["RiskManager", "RiskState", "pip_size"]
