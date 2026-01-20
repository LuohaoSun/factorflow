from typing import Any, Protocol, cast, runtime_checkable

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

try:
    import mlflow
except ImportError:
    mlflow = None

from factorflow.base import Callback, Selector


@runtime_checkable
class FoldEndCallback(Protocol):
    """Protocol for callbacks that implement on_fold_end hook."""

    def on_fold_end(self, selector: Selector, fold: int, logs: dict[str, Any]) -> None:
        """Handle results at the end of each CV fold."""
        ...


# ============================== Helper Functions ==============================


def _compute_classification_metrics(selector: Selector, y_t: np.ndarray, y_p: np.ndarray) -> dict[str, Any]:
    """Compute classification metrics internally."""
    # Ensure correct type for classification
    y_p_final = y_p.astype(y_t.dtype)
    metrics: dict[str, Any] = {"oof_accuracy": float(accuracy_score(y_t, y_p_final))}

    # Confusion Matrix
    labels = np.unique(np.concatenate([y_t, y_p_final]))
    cm = confusion_matrix(y_t, y_p_final, labels=labels)
    metrics["confusion_matrix_df"] = pd.DataFrame(
        cm,
        index=pd.Index([f"Actual_{label}" for label in labels]),
        columns=pd.Index([f"Pred_{label}" for label in labels]),
    )

    # Global OOF AUC
    y_proba_oof = getattr(selector, "y_proba_oof_", None)
    if y_proba_oof is None:
        return metrics

    proba_nan_mask = np.isnan(y_proba_oof).any(axis=1)
    y_t_auc, y_p_auc = y_t[~proba_nan_mask], y_proba_oof[~proba_nan_mask]

    if len(y_t_auc) == 0:
        return metrics

    try:
        if y_p_auc.shape[1] == 2:
            metrics["oof_auc"] = float(roc_auc_score(y_t_auc, y_p_auc[:, 1]))
        else:
            metrics["oof_auc"] = float(roc_auc_score(y_t_auc, y_p_auc, multi_class="ovr", average="macro"))
    except Exception:
        pass

    return metrics


def compute_cv_metrics(selector: Selector, y_true: Any) -> dict[str, Any]:
    """Compute CV metrics from selector results (OOF predictions)."""
    y_pred_oof = getattr(selector, "y_preds_oof_", None)
    task_type = getattr(selector, "task_type", None)

    if y_pred_oof is None or task_type is None:
        return {}

    metrics: dict[str, Any] = {"task_type": task_type}

    # Fold-level average metrics
    fold_auc_scores = getattr(selector, "fold_auc_scores_", [])
    valid_aucs = [a for a in fold_auc_scores if not np.isnan(a)]
    if valid_aucs:
        metrics["cv_avg_auc"] = np.mean(valid_aucs)

    # Filter NaN (failed folds/samples)
    y_true = np.asarray(y_true)
    mask = ~pd.isna(y_pred_oof)
    y_t, y_p = y_true[mask], y_pred_oof[mask]

    if len(y_t) == 0:
        return metrics

    if task_type == "classification":
        metrics.update(_compute_classification_metrics(selector, y_t, y_p))
    else:
        metrics["oof_mae"] = float(mean_absolute_error(y_t, y_p))
        metrics["oof_r2"] = float(r2_score(y_t, y_p))

    return metrics


def create_oof_shap_figures(
    selector: Any,
    X: pd.DataFrame,
    max_display: int = 20,
) -> dict[str, Figure]:
    """Create SHAP summary figures for OOF results."""
    shap_values_oof = getattr(selector, "shap_values_oof_", None)
    if shap_values_oof is None:
        return {}

    valid_mask = ~np.isnan(shap_values_oof).any(axis=1)
    if np.sum(valid_mask) <= 0:
        return {}

    try:
        data_source = getattr(selector, "shap_data_oof_", None)
        base_values_oof = getattr(selector, "base_values_oof_", None)
        n_splits = getattr(selector, "n_splits", "?")
        assert data_source is not None
        assert base_values_oof is not None

        explanation = shap.Explanation(
            values=shap_values_oof[valid_mask],
            data=data_source.iloc[valid_mask].to_numpy(),
            feature_names=data_source.columns.tolist(),
            base_values=base_values_oof[valid_mask] if base_values_oof is not None else None,
        )

        max_name_len = max(len(str(name)) for name in data_source.columns)
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


def create_null_importance_figure(selector: Any, top_k: int = 10) -> Figure | None:
    """Create a figure comparing real vs null importance distributions."""
    import seaborn as sns

    real_importances = getattr(selector, "real_importances_", None)
    if real_importances is None:
        return None

    top_indices = np.argsort(real_importances)[::-1][:top_k]
    if len(top_indices) == 0:
        return None

    try:
        n_cols = 2
        n_rows = (len(top_indices) + 1) // n_cols
        fig = cast(Figure, plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))[0])
        axes = fig.axes

        null_dist = getattr(selector, "null_importances_distribution_", None)
        p_values = getattr(selector, "p_values_", [])
        scores = getattr(selector, "scores_", [])
        feature_names = getattr(selector, "feature_names_in_", [])

        for i, idx in enumerate(top_indices):
            ax = axes[i]
            sns.histplot(null_dist[:, idx] if null_dist is not None else [], color="gray", kde=True, ax=ax)
            ax.axvline(x=float(real_importances[idx]), color="red", linestyle="--", label="Real Imp")
            ax.set_title(f"{feature_names[idx]}\nP-val: {p_values[idx]:.4f}, Ratio: {scores[idx]:.2f}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.warning(f"Failed to create null importance distribution plot: {e}")
        return None


# ============================== Callbacks ==============================


class CVLogger(Callback):
    """Unified callback for logging CV results to console and MLflow."""

    def __init__(self, verbose: bool | int = True, log_mlflow: bool = True, max_display: int = 20):
        """Initialize CVLogger."""
        self.verbose = verbose
        self.log_mlflow = log_mlflow
        self.max_display = max_display

    def on_fit_start(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        """Handle the beginning of fit."""
        if self.verbose:
            logger.info(f"[{selector.label}] Starting {getattr(selector, 'n_splits', '?')}-Fold CV SHAP calculation...")

        if self.log_mlflow and mlflow and mlflow.active_run():
            estimator = getattr(selector, "estimator", None)
            if estimator:
                mlflow.log_param(f"{selector.label}/estimator_type", estimator.__class__.__name__)

    def on_fold_end(self, selector: Selector, fold: int, logs: dict[str, Any]) -> None:
        """Handle the results of each fold."""
        if not self.verbose:
            return

        score, auc = logs.get("score"), logs.get("auc")
        msg = f"Fold {fold + 1}: Score={score:.4f}"
        if auc is not None and not np.isnan(auc):
            msg += f", AUC={auc:.4f}"
        logger.info(msg)

    def on_fit_end(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        """Handle the end of fit (Metrics & Plots)."""
        metrics = compute_cv_metrics(selector, y)
        if not metrics:
            return

        self._log_console(selector, metrics)
        self._log_mlflow(selector, metrics)
        self._log_plots(selector, X)

    def _log_console(self, selector: Selector, metrics: dict[str, Any]):
        if not self.verbose:
            return

        label = selector.label
        if "cv_avg_auc" in metrics:
            logger.info(f"[{label}] Average AUC (per fold): {metrics['cv_avg_auc']:.4f}")

        if metrics["task_type"] == "classification":
            logger.info(f"[{label}] OOF Global Accuracy: {metrics['oof_accuracy']:.4f}")
            if "confusion_matrix_df" in metrics:
                logger.info(f"[{label}] OOF Confusion Matrix:\n{metrics['confusion_matrix_df']}")
            if "oof_auc" in metrics:
                logger.info(f"[{label}] OOF Global AUC: {metrics['oof_auc']:.4f}")
        else:
            logger.info(f"[{label}] OOF Global MAE: {metrics['oof_mae']:.4f}, R2: {metrics['oof_r2']:.4f}")

    def _log_mlflow(self, selector: Selector, metrics: dict[str, Any]):
        if not (self.log_mlflow and mlflow and mlflow.active_run()):
            return

        # Ensure mlflow is not None for type checker
        if mlflow is None:
            return

        label = selector.label
        for k, v in metrics.items():
            if isinstance(v, int | float | np.number):
                mlflow.log_metric(f"{label}/{k}", float(v))

        if "confusion_matrix_df" in metrics:
            mlflow.log_table(
                data=metrics["confusion_matrix_df"].reset_index(),
                artifact_file=f"model/metrics/{label}/oof_confusion_matrix.json",
            )

    def _log_plots(self, selector: Selector, X: pd.DataFrame):
        show_local = self.verbose >= 2
        save_mlflow = self.log_mlflow and mlflow and mlflow.active_run()

        if not (show_local or save_mlflow):
            return

        figures = create_oof_shap_figures(selector, X, max_display=self.max_display)
        if not figures:
            return

        if save_mlflow and mlflow is not None:
            artifact_dir = f"model/plots/{selector.label}"
            for name, fig in figures.items():
                mlflow.log_figure(fig, f"{artifact_dir}/{name}.png")
            logger.info(f"[{selector.label}] SHAP plots recorded to MLflow artifact path: {artifact_dir}")

        if show_local:
            for _fig in figures.values():
                plt.show()

        for fig in figures.values():
            plt.close(fig)


class NullImportanceLogger(Callback):
    """Unified callback for Null Importance logging and plotting."""

    def __init__(self, verbose: bool | int = True, log_mlflow: bool = True, max_display: int = 20):
        """Initialize NullImportanceLogger."""
        self.verbose = verbose
        self.log_mlflow = log_mlflow
        self.max_display = max_display

    def on_fit_end(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        """Handle results of Null Importance selection."""
        self._log_console(selector)
        self._log_mlflow(selector)
        self._log_plots(selector)

    def _log_console(self, selector: Selector):
        if not self.verbose:
            return

        df = pd.DataFrame(
            {
                "feature": selector.feature_names_in_,
                "real_imp": getattr(selector, "real_importances_", []),
                "p_value": getattr(selector, "p_values_", []),
                "ratio": getattr(selector, "scores_", []),
            }
        )
        top_df = df.sort_values(["p_value", "real_imp"], ascending=[True, False]).head(10)
        logger.info(f"[{selector.label}] Null Importance Summary (Top 10):\n{top_df.to_string(index=False)}")

    def _log_mlflow(self, selector: Selector):
        if not (self.log_mlflow and mlflow and mlflow.active_run()):
            return

        if mlflow is not None:
            mlflow.log_metric(f"{selector.label}_selected_count", len(getattr(selector, "selected_features_", [])))

    def _log_plots(self, selector: Selector):
        show_local = self.verbose >= 2
        save_mlflow = self.log_mlflow and mlflow and mlflow.active_run()

        if not (show_local or save_mlflow):
            return

        fig = create_null_importance_figure(selector, top_k=self.max_display)
        if not fig:
            return

        if save_mlflow and mlflow is not None:
            artifact_path = f"model/plots/{selector.label}/null_importance_dist.png"
            mlflow.log_figure(fig, artifact_path)
            logger.info(f"[{selector.label}] Null importance distribution recorded to MLflow: {artifact_path}")

        if show_local:
            plt.show()

        plt.close(fig)
