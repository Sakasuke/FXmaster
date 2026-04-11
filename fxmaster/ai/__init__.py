"""AI engine: training, saving, and loading prediction models.

Note: ``train_model`` is intentionally NOT re-exported here because it pulls in
sklearn and lightgbm. Import it directly from ``fxmaster.ai.trainer`` instead.
"""
from fxmaster.ai.model import TrendModel

__all__ = ["TrendModel"]
