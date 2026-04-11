"""LightGBM-backed trend continuation model.

The model answers one question: given the current feature vector derived from
LTF + HTF candles, what is the probability that a meaningful (>= threshold pips)
price move will happen within the next N bars?

This probability is then fed into the strategy layer which decides whether to
enter, in conjunction with the direction inferred from the HTF trend filter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrendModel:
    """Thin wrapper around a trained LightGBM model and its metadata."""

    booster: Any  # lightgbm.Booster
    feature_names: list[str]
    params: dict
    trained_on: str | None = None
    metrics: dict | None = None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return the probability of trend continuation for each row."""
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns at predict time: {missing}")
        X = features[self.feature_names].to_numpy()
        return np.asarray(self.booster.predict(X))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path, name: str) -> Path:
        import joblib  # lazy import to keep core package dep-free

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{name}.joblib"
        meta_path = directory / f"{name}.json"
        joblib.dump(self.booster, model_path)
        meta_path.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "params": self.params,
                    "trained_on": self.trained_on,
                    "metrics": self.metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return model_path

    @classmethod
    def load(cls, directory: str | Path, name: str) -> "TrendModel":
        import joblib  # lazy import

        directory = Path(directory)
        model_path = directory / f"{name}.joblib"
        meta_path = directory / f"{name}.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Could not find model files for '{name}' in {directory}. "
                "Train the model first with scripts/train_model.py."
            )
        booster = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            booster=booster,
            feature_names=meta["feature_names"],
            params=meta.get("params", {}),
            trained_on=meta.get("trained_on"),
            metrics=meta.get("metrics"),
        )
