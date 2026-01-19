from typing import Any, Literal, cast

from loguru import logger
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

try:
    import mlflow
except ImportError:
    mlflow = None

from factorflow.base import Selector

from ._plot import plot_oof_shap_summary


class SelectFromModelShapCV(Selector):
    """基于交叉验证 SHAP 值的特征选择器 (V2).

    算法原理:
    1. 交叉验证: 通过 K-Fold 交叉验证，每一折使用验证集计算该样本的 SHAP 值。
    2. OOF 拼接: 将所有折验证集的 SHAP 值拼接，形成一个覆盖全量训练数据的 Out-of-Fold (OOF) SHAP 矩阵。
    3. 全局评价: 基于 OOF SHAP 矩阵计算全局特征重要性（均值绝对值），并计算全量数据的预测指标。

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
    feature_importances_: np.ndarray
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

    def _get_selected_features(self) -> list[str]:
        """根据 OOF SHAP 重要性排序返回选中的特征列表."""
        importances = self.feature_importances_
        n_features = len(importances)

        if self.n_features_to_select is None:
            k = None
        elif isinstance(self.n_features_to_select, int):
            k = self.n_features_to_select
        elif isinstance(self.n_features_to_select, float) and 0.0 < self.n_features_to_select < 1.0:
            k = int(n_features * self.n_features_to_select)
        else:
            raise ValueError(f"Invalid n_features_to_select: {self.n_features_to_select}")

        if k is not None:
            if k >= n_features:
                return self.feature_names_in_
            if k <= 0:
                return []

            sorted_indices = np.argsort(importances)
            selected_indices = sorted_indices[-k:]
            return self.get_feature_names_in()[selected_indices].tolist()

        return self.get_feature_names_in()[importances > 0.0].tolist()

    @property
    def feature_names_sorted_(self) -> list[str]:
        """按平均绝对 SHAP 值从大到小排序后的完整特征列表."""
        importances = self.feature_importances_
        indices = np.argsort(importances)[::-1]
        return self.get_feature_names_in()[indices].tolist()

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
        # 统一使用 float 存储预测值，方便使用 np.nan 占位
        self.y_preds_oof_ = np.full(len(y), np.nan, dtype=float)
        self.y_proba_oof_ = None
        self.fold_auc_scores_ = []

        cv = (
            StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            if self.task_type == "classification"
            else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        )

        if self.verbose:
            logger.info(f"[{self.label}] Starting {self.n_splits}-Fold CV SHAP calculation...")

        # 记录模型信息到 MLflow
        if mlflow and mlflow.active_run():
            model_name = self.estimator.__class__.__name__
            mlflow.log_param(f"{self.label}/estimator_type", model_name)

        # 2. 交叉验证循环
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 获取 SHAP 采样索引
            X_shap, fill_idx = self._get_shap_samples(X_val, val_idx, fold_idx)

            # 拟合单折模型
            model = self._fit_fold_model(X_train, y_train, X_val, y_val)

            # 计算这一折的 SHAP 值
            explainer = shap.Explainer(model, X_train)
            explanation = explainer(X_shap)
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
            fold_scores.append(score)

            # 记录概率和 AUC
            auc = self._record_proba_and_auc(model, X_val, y_val, val_idx)

            if self.verbose:
                logger.info(f"Fold {fold_idx + 1}: Score={score:.4f}, AUC={auc:.4f}")

        # 3. 后处理: 计算全局重要性并生成报告
        self.feature_importances_ = np.nanmean(np.abs(self.shap_values_oof_), axis=0)
        self.selected_features_ = self._get_selected_features()

        if self.verbose:
            self._print_global_report(y, fold_scores)

        if self.verbose >= 2:
            plot_oof_shap_summary(self, X, max_display=self.max_display)

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

    def _print_global_report(self, y_true: pd.Series, fold_scores: list[float]):
        """打印全局 OOF 评价报告并记录至 MLflow."""
        avg_score = np.mean(fold_scores) if fold_scores else 0.0
        logger.info(f"[{self.label}] Average Metric (per fold): {avg_score:.4f}")
        valid_aucs = [a for a in self.fold_auc_scores_ if not np.isnan(a)]
        avg_auc = np.mean(valid_aucs) if valid_aucs else None
        if avg_auc is not None:
            logger.info(f"[{self.label}] Average AUC (per fold): {avg_auc:.4f}")

        y_pred_oof = self.y_preds_oof_
        nan_mask = pd.isna(y_pred_oof)
        nan_count = nan_mask.sum()

        if nan_count > 0:
            logger.warning(
                f"[{self.label}] Found {nan_count} samples ({nan_count / len(y_true):.2%}) "
                "without OOF predictions! This usually indicates issues in the CV split or failed folds."
            )

        if self.task_type == "classification":
            mask = ~nan_mask
            y_t, y_p = y_true[mask], y_pred_oof[mask]
            # 关键：将 float 预测值转回真实标签的类型 (如 int)，确保 sklearn 识别为分类任务
            y_p = y_p.astype(y_t.dtype)  # pyright: ignore[reportCallIssue, reportArgumentType]
            acc = float(accuracy_score(y_t, y_p))
            logger.info(f"[{self.label}] OOF Global Accuracy: {acc:.4f}")

            # 混淆矩阵
            labels = np.unique(np.concatenate([y_t, y_p]))
            cm = confusion_matrix(y_t, y_p, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index=pd.Index([f"Actual_{label}" for label in labels]),
                columns=pd.Index([f"Pred_{label}" for label in labels]),
            )
            logger.info(f"[{self.label}] OOF Confusion Matrix:\n{cm_df}")

            if mlflow and mlflow.active_run():
                mlflow.log_metric(f"{self.label}/cv_avg_score", float(avg_score))
                if avg_auc is not None:
                    mlflow.log_metric(f"{self.label}/cv_avg_auc", float(avg_auc))
                mlflow.log_metric(f"{self.label}/oof_accuracy", float(acc))

                mlflow.log_table(
                    data=cm_df.reset_index(),
                    artifact_file=f"model/metrics/{self.label}/oof_confusion_matrix.json",
                )
        else:
            mask = ~np.isnan(y_pred_oof.astype(float))
            y_t, y_p = y_true[mask], y_pred_oof[mask]
            mae = float(mean_absolute_error(y_t, y_p))
            r2 = float(r2_score(y_t, y_p))
            logger.info(f"[{self.label}] OOF Global MAE: {mae:.4f}, R2: {r2:.4f}")
            if mlflow and mlflow.active_run():
                mlflow.log_metric(f"{self.label}/cv_avg_score", float(avg_score))
                mlflow.log_metric(f"{self.label}/oof_mae", float(mae))
                mlflow.log_metric(f"{self.label}/oof_r2", float(r2))

        # 计算 OOF AUC（如果概率可用）
        if self.task_type == "classification" and self.y_proba_oof_ is not None:
            # 过滤掉概率为 NaN 的样本
            proba_nan_mask = np.isnan(self.y_proba_oof_).any(axis=1)
            if proba_nan_mask.any():
                logger.warning(
                    f"[{self.label}] Found {proba_nan_mask.sum()} samples without OOF probabilities! "
                    "OOF AUC will be calculated on valid samples."
                )
                y_true_filtered = y_true[~proba_nan_mask]
                y_proba_filtered = self.y_proba_oof_[~proba_nan_mask]
            else:
                y_true_filtered = y_true
                y_proba_filtered = self.y_proba_oof_

            if len(y_true_filtered) > 0:
                oof_auc = self._calculate_auc(cast(pd.Series, y_true_filtered), y_proba_filtered)
                logger.info(f"[{self.label}] OOF Global AUC: {oof_auc:.4f}")
                if mlflow and mlflow.active_run():
                    mlflow.log_metric(f"{self.label}_oof_auc", float(oof_auc))
