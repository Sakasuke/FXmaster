"""Tests for the risk manager and position sizing math."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fxmaster.config import RiskConfig
from fxmaster.risk.manager import RiskManager


def test_position_size_uses_configured_risk():
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.02), initial_equity=1_000_000.0)
    # On USD/JPY, price distance of 0.20 = 20 pips; units should risk 20,000 JPY
    # (2% of 1,000,000). 20,000 / 0.20 = 100,000 units.
    units = rm.position_size_units("USD_JPY", entry_price=150.00, stop_price=149.80)
    assert units == 100_000


def test_position_size_zero_when_no_stop_distance():
    rm = RiskManager(RiskConfig(), initial_equity=1_000_000.0)
    assert rm.position_size_units("USD_JPY", 150.0, 150.0) == 0


def test_daily_loss_cap_blocks_new_trades():
    cfg = RiskConfig(max_daily_loss_pct=0.06)
    rm = RiskManager(cfg, initial_equity=1_000_000.0)
    now = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    rm.register_open(now=now)
    rm.register_close(-70_000, now=now)  # 7% loss in one shot
    can, reason = rm.can_open_new_position(now=now + timedelta(minutes=1))
    assert not can
    assert "daily" in reason


def test_consecutive_losses_trigger_cooldown():
    cfg = RiskConfig(max_consecutive_losses=3, cooldown_minutes_after_max_losses=30)
    rm = RiskManager(cfg, initial_equity=1_000_000.0)
    now = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    for _ in range(3):
        rm.register_open(now=now)
        rm.register_close(-1000, now=now)
        now = now + timedelta(minutes=1)
    can, reason = rm.can_open_new_position(now=now)
    assert not can
    assert "cooldown" in reason


def test_winning_trade_resets_streak():
    cfg = RiskConfig(max_consecutive_losses=3, cooldown_minutes_after_max_losses=30)
    rm = RiskManager(cfg, initial_equity=1_000_000.0)
    now = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    rm.register_open(now=now)
    rm.register_close(-500, now=now)
    rm.register_open(now=now)
    rm.register_close(+800, now=now)
    assert rm.state.consecutive_losses == 0
