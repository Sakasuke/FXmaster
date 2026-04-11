"""Candle data IO utilities (CSV cache format)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


CANDLE_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def save_candles_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a candle DataFrame to CSV, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(path, index=False)


def load_candles_csv(path: str | Path) -> pd.DataFrame:
    """Load a candle DataFrame from CSV produced by save_candles_csv."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)
