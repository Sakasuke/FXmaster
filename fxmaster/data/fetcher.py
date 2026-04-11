"""OANDA v20 REST API client for fetching candle data.

We intentionally use plain `requests` (no SDK) so that:
  - The dependency surface is minimal
  - It is obvious exactly what HTTPS call is made to OANDA
  - The same client works for demo (practice) and production (live) endpoints.

Reference: https://developer.oanda.com/rest-live-v20/instrument-ep/
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests

logger = logging.getLogger("fxmaster.data.fetcher")


# OANDA supported granularities we expose
Granularity = str  # e.g. "S5", "M1", "M5", "M15", "H1", "H4", "D"


API_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


@dataclass
class OandaCredentials:
    account_id: str
    access_token: str
    environment: str = "practice"

    @property
    def base_url(self) -> str:
        if self.environment not in API_HOSTS:
            raise ValueError(
                f"Unknown OANDA environment '{self.environment}'. Use 'practice' or 'live'."
            )
        return API_HOSTS[self.environment]


_GRANULARITY_SECONDS = {
    "S5": 5,
    "S10": 10,
    "S30": 30,
    "M1": 60,
    "M2": 120,
    "M4": 240,
    "M5": 300,
    "M10": 600,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D": 86400,
    "W": 604800,
}


def granularity_to_seconds(granularity: Granularity) -> int:
    try:
        return _GRANULARITY_SECONDS[granularity]
    except KeyError as exc:
        raise ValueError(f"Unsupported granularity: {granularity}") from exc


class OandaDataFetcher:
    """Thin wrapper around the OANDA v20 `instruments/{instrument}/candles` endpoint.

    The server caps responses to ~5000 candles, so for longer ranges this
    class paginates automatically using the `from`/`to` query parameters.
    """

    MAX_COUNT_PER_REQUEST = 4900  # slightly under the 5000 cap for safety

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

    def fetch_candles(
        self,
        instrument: str,
        granularity: Granularity,
        start: datetime,
        end: datetime | None = None,
        price: str = "M",
    ) -> pd.DataFrame:
        """Fetch candles between ``start`` and ``end`` (both UTC).

        Returns a DataFrame with columns ``time, open, high, low, close, volume``.
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        secs = granularity_to_seconds(granularity)
        window_seconds = secs * self.MAX_COUNT_PER_REQUEST

        all_rows: list[dict] = []
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + timedelta(seconds=window_seconds))
            rows = self._fetch_chunk(instrument, granularity, cursor, chunk_end, price)
            all_rows.extend(rows)
            if not rows:
                cursor = chunk_end
                continue
            last_time = pd.to_datetime(rows[-1]["time"], utc=True).to_pydatetime()
            if last_time <= cursor:
                # Avoid infinite loop in pathological cases
                cursor = chunk_end
            else:
                cursor = last_time + timedelta(seconds=secs)
            # Gentle rate-limit: OANDA allows ~120 req/s but we keep it sane.
            time.sleep(0.1)

        if not all_rows:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        return df

    def _fetch_chunk(
        self,
        instrument: str,
        granularity: Granularity,
        start: datetime,
        end: datetime,
        price: str,
    ) -> list[dict]:
        url = f"{self.credentials.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "price": price,
            "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "includeFirst": "true",
            "smooth": "false",
        }
        logger.debug("GET %s params=%s", url, params)
        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"OANDA candles request failed: {resp.status_code} {resp.text[:300]}"
            )
        payload = resp.json()
        return [_row_from_candle(c) for c in payload.get("candles", []) if c.get("complete")]

    # ------------------------------------------------------------------
    # Live pricing (used by the runner to check current bid/ask & spread)
    # ------------------------------------------------------------------

    def fetch_pricing(self, instruments: Iterable[str]) -> list[dict]:
        url = f"{self.credentials.base_url}/v3/accounts/{self.credentials.account_id}/pricing"
        params = {"instruments": ",".join(instruments)}
        resp = self.session.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(
                f"OANDA pricing request failed: {resp.status_code} {resp.text[:300]}"
            )
        return resp.json().get("prices", [])


def _row_from_candle(candle: dict) -> dict:
    mid = candle.get("mid") or candle.get("bid") or candle.get("ask") or {}
    return {
        "time": candle["time"],
        "open": float(mid["o"]),
        "high": float(mid["h"]),
        "low": float(mid["l"]),
        "close": float(mid["c"]),
        "volume": int(candle.get("volume", 0)),
    }
