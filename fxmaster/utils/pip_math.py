"""Helpers for pip/price conversion that work across JPY and non-JPY pairs."""
from __future__ import annotations


def pip_size(instrument: str) -> float:
    """Return the pip size for the given instrument.

    JPY pairs have a pip of 0.01, everything else 0.0001.
    This matches OANDA convention for major FX pairs.
    """
    return 0.01 if "JPY" in instrument.upper() else 0.0001


def price_to_pips(price_diff: float, instrument: str) -> float:
    return price_diff / pip_size(instrument)


def pips_to_price(pips: float, instrument: str) -> float:
    return pips * pip_size(instrument)
