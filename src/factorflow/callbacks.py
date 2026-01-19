"""Extension callbacks for FactorFlow.

This module is intended for callbacks that require additional dependencies
(e.g., matplotlib, mlflow, shap) or implement specialized logic not suitable
for the core base module.
"""

from .base import Callback, FeatureCheckCallback, FeatureProtectionCallback, Selector, ShapeCheckCallback
from .xai_selectors.callbacks import (
    CVLogger,
    NullImportanceLogger,
)

__all__ = [
    "CVLogger",
    "Callback",
    "FeatureCheckCallback",
    "FeatureProtectionCallback",
    "NullImportanceLogger",
    "Selector",
    "ShapeCheckCallback",
]
