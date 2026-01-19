"""Extension callbacks for FactorFlow.

This module is intended for callbacks that require additional dependencies
(e.g., matplotlib, mlflow, shap) or implement specialized logic not suitable
for the core base module.
"""

from .base import Callback, Selector

__all__ = ["Callback", "Selector"]
