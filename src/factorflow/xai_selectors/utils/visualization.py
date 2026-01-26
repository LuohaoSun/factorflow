from typing import cast

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score


def create_shap_plots(
    shap_values: np.ndarray,
    data: pd.DataFrame,
    base_values: np.ndarray | None,
    n_splits: int | str,
    max_display: int = 20,
) -> dict[str, Figure]:
    """Create SHAP summary figures for OOF results.

    Args:
    ----
        shap_values: SHAP values array.
        data: Feature data corresponding to SHAP values.
        base_values: SHAP base values (expected value).
        n_splits: Number of CV splits (for title).
        max_display: Maximum number of features to display in plots.

    Returns:
    -------
        Dictionary of created matplotlib Figures.
    """
    valid_mask = ~np.isnan(shap_values).any(axis=1)
    if np.sum(valid_mask) <= 0:
        return {}

    try:
        explanation = shap.Explanation(
            values=shap_values[valid_mask],
            data=data.iloc[valid_mask].to_numpy(),
            feature_names=data.columns.tolist(),
            base_values=base_values[valid_mask] if base_values is not None else None,
        )

        max_name_len = max(len(str(name)) for name in data.columns)
        left_margin = min(0.5, 0.05 + max_name_len * 0.008)
        fig_width = max(12, 8 + max_name_len * 0.15)
        fig_height = max(8, max_display * 0.4)

        fig_beeswarm = cast(Figure, plt.figure(figsize=(fig_width, fig_height)))
        shap.plots.beeswarm(explanation, show=False, max_display=max_display)
        plt.title(f"SHAP Beeswarm ({n_splits}-Fold CV)")
        plt.subplots_adjust(left=left_margin, right=0.95, top=0.95, bottom=0.08)

        fig_bar = cast(Figure, plt.figure(figsize=(fig_width, fig_height)))
        shap.plots.bar(explanation, show=False, max_display=max_display)
        plt.title(f"SHAP Bar ({n_splits}-Fold CV)")
        plt.subplots_adjust(left=left_margin, right=0.95, top=0.95, bottom=0.08)

        return {"shap_beeswarm_cv": fig_beeswarm, "shap_bar_cv": fig_bar}
    except Exception as e:
        logger.warning(f"Failed to create SHAP summary plots: {e}")
        return {}


def create_null_importance_plot(
    real_importances: np.ndarray,
    null_dist: np.ndarray,
    feature_names: list[str],
    p_values: np.ndarray,
    scores: np.ndarray,
    top_k: int = 10,
) -> Figure | None:
    """Create a figure comparing real vs null importance distributions."""
    import seaborn as sns

    top_indices = np.argsort(real_importances)[::-1][:top_k]
    if len(top_indices) == 0:
        return None

    try:
        n_cols = 2
        n_rows = (len(top_indices) + 1) // n_cols
        fig = cast(Figure, plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))[0])
        axes = fig.axes

        for i, idx in enumerate(top_indices):
            ax = axes[i]
            sns.histplot(null_dist[:, idx], color="gray", kde=True, ax=ax)
            ax.axvline(x=float(real_importances[idx]), color="red", linestyle="--", label="Real Imp")
            ax.set_title(f"{feature_names[idx]}\nP-val: {p_values[idx]:.4f}, Ratio: {scores[idx]:.2f}")

        for j in range(len(top_indices), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.warning(f"Failed to create null importance distribution plot: {e}")
        return None


def create_regression_plots(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Figure]:
    """Create regression specific plots (Predicted vs Actual, Residuals)."""
    import seaborn as sns

    # Filter NaN
    y_true = np.asarray(y_true)
    mask = ~pd.isna(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]

    if len(y_t) == 0:
        return {}

    figures = {}
    try:
        # Calculate metrics for display
        r2 = r2_score(y_t, y_p)
        mae = mean_absolute_error(y_t, y_p)

        # 1. Predicted vs Actual
        fig_pred = cast(Figure, plt.figure(figsize=(8, 8)))
        ax = fig_pred.add_subplot(111)
        ax.scatter(y_t, y_p, alpha=0.5)

        # Add diagonal line
        min_val = float(np.min([ax.get_xlim(), ax.get_ylim()]))
        max_val = float(np.max([ax.get_xlim(), ax.get_ylim()]))
        lims = [min_val, max_val]

        ax.plot(lims, lims, "r--", alpha=0.75, zorder=0)
        ax.set_aspect("equal")
        ax.set_xlim(left=min_val, right=max_val)
        ax.set_ylim(bottom=min_val, top=max_val)

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Predicted vs Actual\nMAE: {mae:.4f} | R2: {r2:.4f}")
        figures["pred_vs_actual"] = fig_pred

        # 2. Residuals Distribution
        residuals = y_t - y_p
        fig_res = cast(Figure, plt.figure(figsize=(10, 6)))
        ax = fig_res.add_subplot(111)
        sns.histplot(residuals, kde=True, ax=ax)
        ax.axvline(x=0, color="r", linestyle="--")
        ax.set_title("Residuals Distribution")
        ax.set_xlabel("Residual (Actual - Predicted)")
        figures["residuals_dist"] = fig_res

        return figures

    except Exception as e:
        logger.warning(f"Failed to create regression plots: {e}")
        return {}
