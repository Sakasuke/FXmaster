"""Configuration loader for FXmaster."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BrokerConfig:
    name: str = "oanda"
    environment: str = "practice"
    account_id: str = ""
    access_token: str = ""


@dataclass
class TradingHours:
    start: str = "07:00"
    end: str = "21:00"


@dataclass
class TradingConfig:
    instrument: str = "USD_JPY"
    htf_granularity: str = "H1"
    ltf_granularity: str = "M5"
    max_spread_pips: float = 1.5
    trading_hours_utc: TradingHours = field(default_factory=TradingHours)
    news_avoid_minutes_before: int = 15
    news_avoid_minutes_after: int = 30


@dataclass
class StrategyConfig:
    ema_fast: int = 25
    ema_slow: int = 75
    adx_min: float = 25.0
    rsi_period: int = 14
    rsi_pullback_long: float = 40.0
    rsi_pullback_short: float = 60.0
    ai_confidence_min: float = 0.60
    atr_period: int = 14
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 1.8
    partial_take_profit_at_r: float = 1.5
    partial_take_profit_ratio: float = 0.5
    trailing_atr_mult: float = 1.5


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.06
    max_consecutive_losses: int = 4
    cooldown_minutes_after_max_losses: int = 60
    max_concurrent_positions: int = 1


@dataclass
class ModelConfig:
    type: str = "lightgbm"
    params: dict = field(default_factory=dict)
    label_horizon_bars: int = 12
    label_threshold_pips: float = 8.0
    test_size: float = 0.2
    model_dir: str = "data/models"


@dataclass
class DataConfig:
    cache_dir: str = "data/cache"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/fxmaster.log"


@dataclass
class AppConfig:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _coerce(cls, data: dict[str, Any] | None):
    """Build a dataclass instance from a nested dict."""
    if data is None:
        return cls()
    if cls is TradingHours:
        return TradingHours(**data)
    if cls is TradingConfig:
        hours = _coerce(TradingHours, data.get("trading_hours_utc"))
        return TradingConfig(
            instrument=data.get("instrument", "USD_JPY"),
            htf_granularity=data.get("htf_granularity", "H1"),
            ltf_granularity=data.get("ltf_granularity", "M5"),
            max_spread_pips=float(data.get("max_spread_pips", 1.5)),
            trading_hours_utc=hours,
            news_avoid_minutes_before=int(data.get("news_avoid_minutes_before", 15)),
            news_avoid_minutes_after=int(data.get("news_avoid_minutes_after", 30)),
        )
    return cls(**data)


def load_config(path: str | Path) -> AppConfig:
    """Load the YAML config file into an AppConfig object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Copy config/config.example.yaml to config/config.yaml first."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(
        broker=_coerce(BrokerConfig, raw.get("broker")),
        trading=_coerce(TradingConfig, raw.get("trading")),
        strategy=_coerce(StrategyConfig, raw.get("strategy")),
        risk=_coerce(RiskConfig, raw.get("risk")),
        model=_coerce(ModelConfig, raw.get("model")),
        data=_coerce(DataConfig, raw.get("data")),
        logging=_coerce(LoggingConfig, raw.get("logging")),
    )
