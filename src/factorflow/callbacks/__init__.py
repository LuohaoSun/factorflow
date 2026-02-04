"""Extension callbacks for FactorFlow.

This module is intended for callbacks that require additional dependencies
(e.g., matplotlib, mlflow, shap) or implement specialized logic not suitable
for the core base module.
"""

from typing import Any, Protocol, override, runtime_checkable

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from factorflow.base import Callback, CheckFeatures, CheckXShape, ProtectFeatures, Selector


@runtime_checkable
class HasFeatureImportances(Protocol):
    """Protocol for selectors that provide feature importances."""

    feature_importances_: pd.Series | np.ndarray | pd.DataFrame


__all__ = [
    "Callback",
    "CheckFeatures",
    "CheckXShape",
    "LogCorrelationHeatmap",
    "LogXYScatterPlot",
    "LogYDist",
    "ProtectFeatures",
]


def _get_top_features(selector: Selector, top_k: int) -> list[str]:
    """Get top k features based on feature_importances_ or selection order."""
    selected_features = selector.selected_features_
    if not selected_features:
        return []

    # Try to use feature_importances_ if available
    feature_importances_ = getattr(selector, "feature_importances_", None)
    if feature_importances_ is not None:
        # feature_importances_ should correspond to feature_names_in_
        importances = feature_importances_
        feature_names = selector.feature_names_in_

        # Create a series for easy filtering and sorting
        importance_series = pd.Series(importances, index=feature_names)
        # Only keep selected features
        importance_series = importance_series.loc[selected_features]
        # Sort by importance descending
        top_features = importance_series.sort_values(ascending=False).head(top_k).index.tolist()
    else:
        # Fallback to selection order
        top_features = selected_features[:top_k]

    return top_features


class LogXYScatterPlot(Callback):
    """Plot scatter plots for top features vs target."""

    def __init__(self, top_k: int = 10, alpha: float = 0.5):
        """Initialize LogFeatureTargetScatterPlot.

        Args:
        ----
            top_k: Number of top features to plot.
            alpha: Transparency of scatter points.
        """
        self.top_k = top_k
        self.alpha = alpha

    @override
    def on_callback_add(self, selector: Selector) -> None:
        """Check if selector has feature_importances_."""
        if not isinstance(selector, HasFeatureImportances):
            logger.warning(
                f"[{selector.__class__.__name__}] LogXYScatterPlot requires feature_importances_ "
                "for top_k feature selection. Falling back to selection order."
            )

    @override
    def on_fit_end(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        """Plot scatter plots after fit."""
        if y is None:
            return

        top_features = _get_top_features(selector, self.top_k)
        if not top_features:
            return

        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        # Handle axes flattening for different subplot configurations
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, col in enumerate(top_features):
            sns.scatterplot(x=X[col], y=y, alpha=self.alpha, ax=axes_flat[i])
            axes_flat[i].set_title(f"{col} vs Target")

        # Hide unused axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()
        if mlflow.active_run():
            logger.info(f"[{selector.label}] Logging feature target scatter plot to MLflow")
            mlflow.log_figure(
                fig,
                f"feature_target_scatter_{selector.label}.png",
                save_kwargs={"bbox_inches": "tight"},
            )
        else:
            plt.show(fig)
        plt.close(fig)


class LogCorrelationHeatmap(Callback):
    """Plot correlation heatmap for top features."""

    def __init__(self, top_k: int = 20, cmap: str = "coolwarm", include_y: bool = False):
        """Initialize LogCorrelationHeatmap.

        Args:
        ----
            top_k: Number of top features to include in the heatmap.
            cmap: Colormap for the heatmap.
            include_y: Whether to include the target variable y in the correlation matrix.
        """
        self.top_k = top_k
        self.cmap = cmap
        self.include_y = include_y

    @override
    def on_callback_add(self, selector: Selector) -> None:
        """Check if selector has feature_importances_."""
        if not isinstance(selector, HasFeatureImportances):
            logger.warning(
                f"[{selector.__class__.__name__}] LogCorrelationHeatmap requires feature_importances_ "
                "for top_k feature selection. Falling back to selection order."
            )

    @override
    def on_fit_end(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        """Plot correlation heatmap after fit."""
        top_features = _get_top_features(selector, self.top_k)
        if not top_features:
            return

        data_to_corr = X[top_features].copy()
        if self.include_y and y is not None:
            y_name = "target"
            if hasattr(y, "name") and y.name:
                y_name = y.name
            elif isinstance(y, pd.DataFrame) and len(y.columns) > 0:
                y_name = y.columns[0]

            # Ensure y is a series or array-like that can be added to DataFrame
            data_to_corr[y_name] = y

        corr = data_to_corr.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap=self.cmap, center=0, ax=ax)
        ax.set_title(f"Correlation Heatmap (Top {len(top_features)})")

        plt.tight_layout()
        if mlflow.active_run():
            logger.info(f"[{selector.label}] Logging correlation heatmap to MLflow")
            mlflow.log_figure(
                fig,
                f"correlation_heatmap_{selector.label}.png",
                save_kwargs={"bbox_inches": "tight"},
            )
        else:
            plt.show(fig)
        plt.close(fig)


class LogYDist(Callback):
    """Plot distribution of the target variable."""

    @override
    def on_fit_start(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        """Plot target distribution before fit."""
        if y is None:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(y, kde=True, ax=ax)
        ax.set_title("Target Distribution")

        plt.tight_layout()
        if mlflow.active_run():
            logger.info(f"[{selector.label}] Logging target distribution to MLflow")
            mlflow.log_figure(
                fig,
                f"target_distribution_{selector.label}.png",
                save_kwargs={"bbox_inches": "tight"},
            )
        else:
            plt.show(fig)
        plt.close(fig)
