"""Abstract broker interface.

Concrete implementations must translate the generic orders below into their
native API (OANDA v20, MT5, etc.). Keeping this thin means the rest of the
system does not care which venue we are trading at.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AccountInfo:
    account_id: str
    currency: str
    balance: float
    equity: float


@dataclass
class OrderResult:
    order_id: str
    instrument: str
    units: int
    price: float
    stop_loss: float | None
    take_profit: float | None
    raw: dict


@dataclass
class OpenPosition:
    instrument: str
    units: int
    average_price: float
    unrealized_pnl: float


class BrokerInterface(ABC):
    @abstractmethod
    def account_info(self) -> AccountInfo: ...

    @abstractmethod
    def get_price(self, instrument: str) -> tuple[float, float]:
        """Return (bid, ask)."""

    @abstractmethod
    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    def close_position(self, instrument: str) -> None: ...

    @abstractmethod
    def get_open_position(self, instrument: str) -> OpenPosition | None: ...
