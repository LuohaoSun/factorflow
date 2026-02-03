from typing import Any, Literal, Protocol, cast, runtime_checkable

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

try:
    import mlflow
except ImportError:
    mlflow = None

from factorflow.base import Callback, Selector
from factorflow.xai_selectors.utils import metrics, visualization


@runtime_checkable
class FoldEndCallback(Protocol):
    """Protocol for callbacks that implement on_fold_end hook."""

    def on_fold_end(self, selector: Selector, fold: int, logs: dict[str, Any]) -> None:
        """Handle results at the end of each CV fold."""
        ...


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
            n_splits = getattr(selector, "n_splits", "?")
            logger.info(f"[{selector.label}] Starting {n_splits}-Fold CV SHAP calculation...")

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
        # Only compute metrics if it's a CV selector (has OOF preds)
        if not hasattr(selector, "y_preds_oof_"):
            return

        # Safe cast as we are inside the class definition file
        cv_selector = cast("SelectFromModelShapCV", selector)
        m = metrics.compute_cv_metrics(cv_selector, y)
        if not m:
            return

        self._log_console(cv_selector, m)
        self._log_mlflow(cv_selector, m)
        self._log_plots(cv_selector, X, y)

    def _log_console(self, selector: "SelectFromModelShapCV", metrics_dict: dict[str, Any]):
        if not self.verbose:
            return

        label = selector.label
        if "cv_avg_auc" in metrics_dict:
            logger.info(f"[{label}] Average AUC (per fold): {metrics_dict['cv_avg_auc']:.4f}")

        if metrics_dict["task_type"] == "classification":
            logger.info(f"[{label}] OOF Global Accuracy: {metrics_dict['oof_accuracy']:.4f}")
            if "confusion_matrix_df" in metrics_dict:
                logger.info(f"[{label}] OOF Confusion Matrix:\n{metrics_dict['confusion_matrix_df']}")
            if "oof_auc" in metrics_dict:
                logger.info(f"[{label}] OOF Global AUC: {metrics_dict['oof_auc']:.4f}")
        else:
            logger.info(f"[{label}] OOF Global MAE: {metrics_dict['oof_mae']:.4f}, R2: {metrics_dict['oof_r2']:.4f}")

    def _log_mlflow(self, selector: "SelectFromModelShapCV", metrics_dict: dict[str, Any]):
        if not (self.log_mlflow and mlflow and mlflow.active_run()):
            return

        # Ensure mlflow is not None for type checker
        if mlflow is None:
            return

        label = selector.label
        for k, v in metrics_dict.items():
            if isinstance(v, int | float | np.number):
                mlflow.log_metric(f"{label}/{k}", float(v))

        if "confusion_matrix_df" in metrics_dict:
            mlflow.log_table(
                data=metrics_dict["confusion_matrix_df"].reset_index(),
                artifact_file=f"model/metrics/{label}/oof_confusion_matrix.json",
            )

    def _log_plots(self, selector: "SelectFromModelShapCV", X: pd.DataFrame, y: Any = None):
        show_local = self.verbose >= 2
        save_mlflow = self.log_mlflow and mlflow and mlflow.active_run()

        if not (show_local or save_mlflow):
            return

        figures = {}

        # 1. SHAP Plots
        if selector.shap_values_oof_ is not None and selector.shap_data_oof_ is not None:
            figures.update(
                visualization.create_shap_plots(
                    shap_values=selector.shap_values_oof_,
                    data=selector.shap_data_oof_,
                    base_values=selector.base_values_oof_,
                    n_splits=getattr(selector, "n_splits", "?"),
                    max_display=self.max_display,
                )
            )

        # 2. Regression Plots
        task_type = getattr(selector, "task_type", None)
        y_pred_oof = getattr(selector, "y_preds_oof_", None)
        if task_type != "classification" and y is not None and y_pred_oof is not None:
            figures.update(visualization.create_regression_plots(y_true=y, y_pred=y_pred_oof))

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


class SelectFromModelShapCV(Selector):
    """基于交叉验证 SHAP 值的特征选择器 (V2).

    算法原理:
    1. 交叉验证: 通过 K-Fold 交叉验证，每一折使用验证集计算该样本的 SHAP 值。
    2. OOF 拼接: 将所有折验证集的 SHAP 值拼接，形成一个覆盖全量训练数据的 Out-of-Fold (OOF) SHAP 矩阵。
    3. 全局评价: 基于 OOF SHAP 矩阵计算全局特征重要性（均值绝对值）。

    该方法比单次 Hold-out 验证更稳健，因为它利用了所有训练样本来评估特征贡献。
    所有 OOF 属性的样本顺序与输入 X 的原始样本顺序保持一致。

    属性:
        shap_values_oof_ (np.ndarray): 全量数据的 OOF SHAP 值矩阵. 与 X 的形状和顺序一致
        base_values_oof_ (np.ndarray): 每个样本的 SHAP 基准值 (Expected Value). 与 X 的形状和顺序一致
        shap_data_oof_ (pd.DataFrame): 与 SHAP 对齐的原始特征值（用于绘图）. 就是 X, 仅当 store_shap_data=True 时存在.
        y_preds_oof_ (np.ndarray): 全量数据的 OOF 预测值. 与 X 的形状和顺序一致
        y_proba_oof_ (np.ndarray): 全量数据的 OOF 预测概率 (仅分类任务). 与 X 的形状和顺序一致
        feature_importances_ (np.ndarray): 全局特征重要性（OOF SHAP 绝对值的均值）.
        feature_names_sorted_ (list[str]): 按平均绝对 SHAP 值从大到小排序后的完整特征列表.
        fold_auc_scores_ (list): 每一折的 AUC 得分.
    """

    shap_values_oof_: np.ndarray
    base_values_oof_: np.ndarray
    shap_data_oof_: pd.DataFrame | None
    y_preds_oof_: np.ndarray
    y_proba_oof_: np.ndarray | None
    feature_importances_: pd.Series = None  # type: ignore
    fold_auc_scores_: list[float]

    def __init__(
        self,
        estimator: BaseEstimator,
        task_type: Literal["classification", "regression"],
        n_features_to_select: int | float | None = None,
        n_splits: int = 5,
        random_state: int = 42,
        shap_sample_size: int | None = None,
        fit_uses_eval_set: bool = False,
        verbose: bool | int = True,
        max_display: int = 20,
        model_fit_params: dict[str, Any] | None = None,
        store_shap_data: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化 CV SHAP 选择器.

        Args:
        ----
            estimator: 基础估计器.
            task_type: 任务类型, "classification" 或 "regression".
            n_features_to_select: 要选择的特征数量. 整数表示个数, 浮点数 (0-1) 表示比例,
                None 则保留重要性 > 0 的所有特征.
            n_splits: 交叉验证折数.
            random_state: 随机种子.
            shap_sample_size: 每一折计算 SHAP 时的采样数量 (为加速计算).
            fit_uses_eval_set: 是否在 fit 时传入验证集作为 eval_set.
            verbose: 日志详尽程度. 0: 静默; 1: 打印指标; 2: 打印并本地绘图.
            max_display: 绘图显示的 Top 特征数.
            model_fit_params: 透传给 fit 的额外参数.
            store_shap_data: 是否存储 X 的全量数据副本以便后续绘图.
            **kwargs: 透传给父类 BaseSelector 的参数.
        """
        # 初始化统一的 CVLogger

        super().__init__(**kwargs)
        self.estimator = estimator
        self.task_type = task_type
        self.n_features_to_select = n_features_to_select
        self.n_splits = n_splits
        self.random_state = random_state
        self.shap_sample_size = shap_sample_size
        self.fit_uses_eval_set = fit_uses_eval_set
        self.verbose = verbose
        self.max_display = max_display
        self.model_fit_params = model_fit_params
        self.store_shap_data = store_shap_data
        self.add_callback(CVLogger(verbose=verbose, max_display=max_display))

    def _get_selected_features(self) -> list:
        """根据 OOF SHAP 重要性排序返回选中的特征列表."""
        if self.n_features_to_select is None:
            # 保留重要性 > 0 的特征
            return self.feature_importances_[self.feature_importances_ > 0].index.tolist()  # pyright: ignore[reportAttributeAccessIssue]

        n_features = len(self.feature_importances_)
        k = self.n_features_to_select

        if isinstance(k, float) and 0.0 < k < 1.0:
            k = int(n_features * k)
        elif not isinstance(k, int):
            raise ValueError(f"Invalid n_features_to_select: {k}")

        if k >= n_features:
            return self.feature_importances_.index.tolist()
        if k <= 0:
            return []

        # feature_importances_ 保持原始顺序，使用 nlargest 筛选
        return self.feature_importances_.nlargest(k).index.tolist()

    @property
    def feature_names_sorted_(self) -> list:
        """按平均绝对 SHAP 值从大到小排序后的完整特征列表."""
        return self.feature_importances_.sort_values(ascending=False).index.tolist()

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs: Any) -> "SelectFromModelShapCV":
        """执行交叉验证并收集全量 OOF SHAP 值."""
        if not np.issubdtype(np.asarray(y).dtype, np.number):
            raise ValueError(
                f"Target variable y must be numeric for SHAP selectors, but got dtype: {np.asarray(y).dtype}"
            )

        # 1. 初始化容器
        self.shap_values_oof_ = np.full((X.shape[0], X.shape[1]), np.nan)
        self.base_values_oof_ = np.full(X.shape[0], np.nan)
        self.shap_data_oof_ = (
            pd.DataFrame(index=X.index, columns=X.columns, dtype=float) if self.store_shap_data else None
        )
        self.y_preds_oof_ = np.full(len(y), np.nan, dtype=float)
        self.y_proba_oof_ = None
        self.fold_auc_scores_ = []

        cv = (
            StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            if self.task_type == "classification"
            else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        )

        # 2. 交叉验证循环
        y_series = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]

            # 获取 SHAP 采样索引
            X_shap, fill_idx = self._get_shap_samples(X_val, val_idx, fold_idx)

            # 拟合单折模型
            model = self._fit_fold_model(X_train, y_train, X_val, y_val)

            # 计算这一折的 SHAP 值
            explainer = shap.Explainer(model, X_train)
            explanation = explainer(X_shap, check_additivity=False)
            shap_vals, base_vals = self._process_shap_output(explanation)

            # 存储至 OOF 矩阵
            self.shap_values_oof_[fill_idx] = shap_vals
            self.base_values_oof_[fill_idx] = np.asarray(base_vals).reshape(-1)
            if self.shap_data_oof_ is not None:
                self.shap_data_oof_.loc[X_shap.index] = X_shap

            # 计算并存储预测值
            y_pred = model.predict(X_val)
            self.y_preds_oof_[val_idx] = y_pred

            score = self._evaluate_fold_metric(y_val, y_pred)
            auc = self._record_proba_and_auc(model, X_val, y_val, val_idx)

            # 触发单折回调 (on_fold_end)
            fold_logs = {"score": score, "auc": auc}
            for cb in self._callbacks:
                if isinstance(cb, FoldEndCallback):
                    cb.on_fold_end(self, fold_idx, fold_logs)

        # 3. 后处理: 计算全局重要性
        importances_arr = np.nanmean(np.abs(self.shap_values_oof_), axis=0)
        self.feature_importances_ = pd.Series(importances_arr, index=self.feature_names_in_)
        self.selected_features_ = self._get_selected_features()

        return self

    def _get_shap_samples(
        self,
        X_val: pd.DataFrame,
        val_idx: np.ndarray,
        fold_idx: int,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """获取当前折用于 SHAP 计算的样本及其索引."""
        if not self.shap_sample_size or len(X_val) <= self.shap_sample_size:
            return X_val, val_idx
        rng = np.random.RandomState(self.random_state + fold_idx)
        local_idx = rng.choice(len(X_val), self.shap_sample_size, replace=False)
        return X_val.iloc[local_idx], val_idx[local_idx]

    def _fit_fold_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """克隆并训练单折模型."""
        model: Any = clone(self.estimator)
        fit_params = (self.model_fit_params or {}).copy()
        if self.fit_uses_eval_set:
            fit_params.update({"eval_set": [(X_val, y_val)], "verbose": False})
        model.fit(X_train, y_train, **fit_params)
        return model

    def _process_shap_output(self, explanation: Any) -> tuple[np.ndarray, np.ndarray]:
        """标准化 SHAP 输出, 自动处理多分类降维."""
        shap_vals = explanation.values  # noqa: PD011
        base_vals = explanation.base_values
        if shap_vals.ndim == 3:  # (samples, features, classes)
            shap_vals = np.abs(shap_vals).mean(axis=-1)
            base_vals = base_vals[:, 0] if base_vals.ndim > 1 else base_vals
        return shap_vals, base_vals

    def _record_proba_and_auc(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series, val_idx: np.ndarray) -> float:
        """记录验证集概率并计算单折 AUC."""
        if self.task_type != "classification" or not hasattr(model, "predict_proba"):
            self.fold_auc_scores_.append(np.nan)
            return np.nan
        y_proba = model.predict_proba(X_val)
        if self.y_proba_oof_ is None:
            self.y_proba_oof_ = np.full((len(self.y_preds_oof_), y_proba.shape[1]), np.nan)
        self.y_proba_oof_[val_idx] = y_proba
        auc = self._calculate_auc(y_val, y_proba)
        self.fold_auc_scores_.append(auc)
        return auc

    def _evaluate_fold_metric(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """计算单折的核心指标 (Accuracy 或 -MAE)."""
        if self.task_type == "classification":
            return float(accuracy_score(y_true, y_pred))
        return float(-mean_absolute_error(y_true, y_pred))

    def _calculate_auc(self, y_true: pd.Series, y_proba: np.ndarray) -> float:
        """稳健地计算 AUC 指标."""
        try:
            if y_proba.shape[1] == 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            return np.nan
