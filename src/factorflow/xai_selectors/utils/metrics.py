from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    from factorflow.xai_selectors.select_from_model_shap_cv import SelectFromModelShapCV

    # Only CV selector has OOF predictions needed for metrics
    CVSelector = SelectFromModelShapCV
else:
    CVSelector = Any


def _compute_classification_metrics(selector: CVSelector, y_t: np.ndarray, y_p: np.ndarray) -> dict[str, Any]:
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
    y_proba_oof = selector.y_proba_oof_
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


def compute_cv_metrics(selector: CVSelector, y_true: Any) -> dict[str, Any]:
    """Compute CV metrics from selector results (OOF predictions)."""
    # Safe access for static analysis, though runtime calls should ensure selector is CV type
    y_pred_oof = getattr(selector, "y_preds_oof_", None)
    task_type = getattr(selector, "task_type", None)

    if y_pred_oof is None or task_type is None:
        return {}

    metrics: dict[str, Any] = {"task_type": task_type}

    # Fold-level average metrics
    # Note: Using getattr to handle list default safely if attribute doesn't exist (though Protocol says it should)
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
