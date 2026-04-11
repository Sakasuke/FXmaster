"""OANDA v20 REST broker implementation.

Covers the minimum surface needed to run the strategy:
  * Account summary (balance / currency)
  * Pricing (bid/ask)
  * Market order with attached SL/TP (one-cancels-other style)
  * Position inspection & close

All requests use the same base URL as the data fetcher so the environment
(practice vs live) flows from the same credential object.
"""
from __future__ import annotations

import logging

import requests

from fxmaster.broker.base import AccountInfo, BrokerInterface, OpenPosition, OrderResult
from fxmaster.data.fetcher import OandaCredentials

logger = logging.getLogger("fxmaster.broker.oanda")


class OandaBroker(BrokerInterface):
    def __init__(self, credentials: OandaCredentials, session: requests.Session | None = None):
        self.credentials = credentials
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {credentials.access_token}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            }
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def account_info(self) -> AccountInfo:
        url = f"{self.credentials.base_url}/v3/accounts/{self.credentials.account_id}/summary"
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("account", {})
        return AccountInfo(
            account_id=data.get("id", ""),
            currency=data.get("currency", ""),
            balance=float(data.get("balance", 0.0)),
            equity=float(data.get("NAV", data.get("balance", 0.0))),
        )

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def get_price(self, instrument: str) -> tuple[float, float]:
        url = f"{self.credentials.base_url}/v3/accounts/{self.credentials.account_id}/pricing"
        resp = self.session.get(url, params={"instruments": instrument}, timeout=10)
        resp.raise_for_status()
        prices = resp.json().get("prices", [])
        if not prices:
            raise RuntimeError(f"No price returned for {instrument}")
        p = prices[0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        return bid, ask

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        if units == 0:
            raise ValueError("Refusing to place an order with 0 units")

        url = f"{self.credentials.base_url}/v3/accounts/{self.credentials.account_id}/orders"
        order: dict = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if stop_loss is not None:
            order["stopLossOnFill"] = {"price": f"{stop_loss:.5f}", "timeInForce": "GTC"}
        if take_profit is not None:
            order["takeProfitOnFill"] = {"price": f"{take_profit:.5f}", "timeInForce": "GTC"}

        logger.info("Placing market order: %s", order)
        resp = self.session.post(url, json={"order": order}, timeout=20)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"OANDA order placement failed: {resp.status_code} {resp.text[:300]}"
            )
        payload = resp.json()
        fill = payload.get("orderFillTransaction") or {}
        return OrderResult(
            order_id=str(fill.get("id", "")),
            instrument=instrument,
            units=int(fill.get("units", units)),
            price=float(fill.get("price", 0.0)),
            stop_loss=stop_loss,
            take_profit=take_profit,
            raw=payload,
        )

    def close_position(self, instrument: str) -> None:
        url = (
            f"{self.credentials.base_url}/v3/accounts/{self.credentials.account_id}"
            f"/positions/{instrument}/close"
        )
        resp = self.session.put(
            url,
            json={"longUnits": "ALL", "shortUnits": "ALL"},
            timeout=20,
        )
        # OANDA returns 200 on success, 404 if there is no position to close.
        if resp.status_code not in (200, 201, 404):
            raise RuntimeError(
                f"OANDA close position failed: {resp.status_code} {resp.text[:300]}"
            )

    def get_open_position(self, instrument: str) -> OpenPosition | None:
        url = (
            f"{self.credentials.base_url}/v3/accounts/{self.credentials.account_id}"
            f"/positions/{instrument}"
        )
        resp = self.session.get(url, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        position = resp.json().get("position", {})
        long_units = int(position.get("long", {}).get("units", 0) or 0)
        short_units = int(position.get("short", {}).get("units", 0) or 0)
        units = long_units + short_units
        if units == 0:
            return None
        if long_units != 0:
            avg = float(position["long"].get("averagePrice", 0.0))
            upnl = float(position["long"].get("unrealizedPL", 0.0))
        else:
            avg = float(position["short"].get("averagePrice", 0.0))
            upnl = float(position["short"].get("unrealizedPL", 0.0))
        return OpenPosition(
            instrument=instrument,
            units=units,
            average_price=avg,
            unrealized_pnl=upnl,
        )
