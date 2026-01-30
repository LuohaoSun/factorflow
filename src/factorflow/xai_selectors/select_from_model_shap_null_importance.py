from typing import Any, Literal, cast

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

try:
    import mlflow
except ImportError:
    mlflow = None

from factorflow.base import Callback, Selector
from factorflow.xai_selectors.select_from_model_shap_cv import SelectFromModelShapCV
from factorflow.xai_selectors.utils import visualization


class NullImportanceLogger(Callback):
    """Unified callback for Null Importance logging and plotting."""

    def __init__(self, verbose: bool | int = True, log_mlflow: bool = True, max_display: int = 20):
        """Initialize NullImportanceLogger."""
        self.verbose = verbose
        self.log_mlflow = log_mlflow
        self.max_display = max_display

    def on_fit_end(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        """Handle results of Null Importance selection."""
        # Only process if selector has null importance attributes
        if not hasattr(selector, "real_importances_"):
            return

        null_selector = cast("SelectFromModelShapNullImportance", selector)
        self._log_console(null_selector)
        self._log_mlflow(null_selector)
        self._log_plots(null_selector)

    def _log_console(self, selector: "SelectFromModelShapNullImportance"):
        if not self.verbose:
            return

        # Safe access using specific type
        real_imp = selector.real_importances_
        p_vals = selector.p_values_
        scores = selector.scores_
        features = selector.feature_names_in_

        if real_imp is None or p_vals is None or scores is None:
            return

        df = pd.DataFrame(
            {
                "feature": features,
                "real_imp": real_imp,
                "p_value": p_vals,
                "ratio": scores,
            }
        )
        top_df = df.sort_values(["p_value", "real_imp"], ascending=[True, False]).head(10)
        logger.info(f"[{selector.label}] Null Importance Summary (Top 10):\n{top_df.to_string(index=False)}")

    def _log_mlflow(self, selector: "SelectFromModelShapNullImportance"):
        if not (self.log_mlflow and mlflow and mlflow.active_run()):
            return

        if mlflow is not None:
            # Note: selector.selected_features_ comes from base Selector
            mlflow.log_metric(f"{selector.label}_selected_count", len(selector.selected_features_))

    def _log_plots(self, selector: "SelectFromModelShapNullImportance"):
        show_local = self.verbose >= 2
        save_mlflow = self.log_mlflow and mlflow and mlflow.active_run()

        if not (show_local or save_mlflow):
            return

        real_imp = selector.real_importances_
        null_dist = selector.null_importances_distribution_
        if real_imp is None or null_dist is None or selector.p_values_ is None or selector.scores_ is None:
            return

        fig = visualization.create_null_importance_plot(
            real_importances=real_imp,
            null_dist=null_dist,
            feature_names=selector.feature_names_in_.tolist(),
            p_values=selector.p_values_,
            scores=selector.scores_,
            top_k=self.max_display,
        )

        if not fig:
            return

        if save_mlflow and mlflow is not None:
            artifact_path = f"model/plots/{selector.label}/null_importance_dist.png"
            mlflow.log_figure(fig, artifact_path)
            logger.info(f"[{selector.label}] Null importance distribution recorded to MLflow: {artifact_path}")

        if show_local:
            plt.show()

        plt.close(fig)


class SelectFromModelShapNullImportance(Selector):
    """基于 Null Importance (目标变量置换) 的 SHAP 特征选择器 (V2).

    算法原理:
    1. Base Run: 在原始数据上使用交叉验证计算稳健的 SHAP 特征重要性（真实重要性）。
    2. Null Runs: 多次随机打乱目标变量 y 的顺序（Target Permutation），并重新训练模型计算 SHAP 重要性。
       这一步构建了每个特征在“无意义数据”上的重要性零分布 (Null Distribution)。
    3. 显著性评估:
       - P-value: 计算零分布中重要性大于或等于真实重要性的次数占比。
       - Ratio: 计算真实重要性与零分布均值的比率。

    属性:
        real_importances_ (np.ndarray): 原始数据上的特征重要性 (Base Run).
        null_importances_distribution_ (np.ndarray): 零分布矩阵，形状为 (n_trials, n_features).
        p_values_ (np.ndarray): 每个特征的经验 P 值.
        scores_ (np.ndarray): 根据 mode 计算的得分 (P-value 或 Ratio).
        shap_values_ (np.ndarray): Base Run 产生的 OOF SHAP 值矩阵.
    """

    real_importances_: np.ndarray
    null_importances_distribution_: np.ndarray
    p_values_: np.ndarray
    scores_: np.ndarray
    shap_values_: np.ndarray
    feature_importances_: pd.Series

    def __init__(
        self,
        estimator: BaseEstimator,
        task_type: Literal["classification", "regression"],
        n_trials: int = 50,
        mode: Literal["p_value", "ratio"] = "p_value",
        threshold: float = 0.05,
        val_size: float = 0.2,
        random_state: int = 42,
        shap_sample_size: int | None = None,
        fit_uses_eval_set: bool = False,
        verbose: bool | int = True,
        max_display: int = 20,
        model_fit_params: dict[str, Any] | None = None,
        cv: int = 5,
        **kwargs: Any,
    ) -> None:
        """初始化 Null Importance 选择器."""
        # 设置默认 Callbacks
        default_callbacks: list[Callback] = [
            NullImportanceLogger(verbose=verbose, max_display=max_display),
        ]
        user_callbacks: list[Callback] = kwargs.pop("callbacks", []) or []
        all_callbacks = default_callbacks + user_callbacks

        super().__init__(callbacks=all_callbacks, **kwargs)
        self.estimator = estimator
        self.task_type = task_type
        self.n_trials = n_trials
        self.mode = mode
        self.threshold = threshold
        self.val_size = val_size
        self.random_state = random_state
        self.shap_sample_size = shap_sample_size
        self.fit_uses_eval_set = fit_uses_eval_set
        self.verbose = verbose
        self.max_display = max_display
        self.model_fit_params = model_fit_params
        self.cv = cv

    def _get_selected_features(self) -> list[str]:
        """根据显著性测试结果返回选中的特征名称列表."""
        if self.mode == "p_value":
            mask = self.p_values_ < self.threshold
        elif self.mode == "ratio":
            mask = self.scores_ > self.threshold
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return np.array(self.feature_names_in_)[mask].tolist()

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs: Any) -> "SelectFromModelShapNullImportance":
        """执行特征选择逻辑."""
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(f"Target variable y must be numeric for SHAP selectors, but got {y.dtype}")

        # 1. Base Run
        if self.verbose:
            logger.info(f"[{self.label}] Phase 1: Calculating Base Importance (Real Data) with {self.cv}-Fold CV...")

        base_selector = SelectFromModelShapCV(
            estimator=self.estimator,
            task_type=cast(Literal["classification", "regression"], self.task_type),
            n_splits=self.cv,
            random_state=self.random_state,
            shap_sample_size=self.shap_sample_size,
            fit_uses_eval_set=self.fit_uses_eval_set,
            verbose=self.verbose,
            model_fit_params=self.model_fit_params,
            store_shap_data=False,
            label=f"{self.label}_BaseRun",
        )
        base_selector.fit(X, y)

        self.shap_values_ = base_selector.shap_values_oof_
        # base_selector.feature_importances_ 是 Series, 取 values 以保持 numpy 兼容性
        # 注意: Series 已经排过序了, 但 feature_names_in_ 是原始顺序
        # 所以必须用 loc[] 重新对齐到 feature_names_in_ 的顺序
        self.real_importances_ = base_selector.feature_importances_.loc[self.feature_names_in_].to_numpy()

        # 2. Null Runs
        null_imps_list = []
        rng = np.random.RandomState(self.random_state)

        if self.verbose:
            logger.info(f"[{self.label}] Phase 2: Running {self.n_trials} Null Trials (Target Permutation)...")

        iterator = range(self.n_trials)
        if self.verbose:
            iterator = tqdm(iterator, desc="Null Trials")

        for _ in iterator:
            # 标签打乱
            y_permuted = rng.permutation(y)
            y_permuted_series = pd.Series(y_permuted, index=X.index)

            # Null Run 采用单次切分以兼顾效率与有效性
            X_train, X_val, y_train, y_val = cast(
                tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                train_test_split(X, y_permuted_series, test_size=self.val_size, random_state=rng),
            )

            model = self._fit_null_model(X_train, y_train, X_val, y_val)

            # 计算 SHAP
            X_shap_val = X_val
            if self.shap_sample_size and len(X_val) > self.shap_sample_size:
                X_shap_val = X_val.sample(n=self.shap_sample_size, random_state=rng)

            explainer = shap.Explainer(model, X_train)
            explanation = explainer(X_shap_val, check_additivity=False)

            shap_vals, _ = self._process_shap_output(explanation)
            null_imp = np.abs(shap_vals).mean(axis=0)
            null_imps_list.append(null_imp)

        self.null_importances_distribution_ = np.array(null_imps_list)

        # 3. 统计指标计算
        self._calculate_stats()
        self.selected_features_ = self._get_selected_features()

        return self

    def _fit_null_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """为 Null Trial 拟合一个独立的模型克隆."""
        model: Any = clone(self.estimator)
        fit_params = (self.model_fit_params or {}).copy()
        if self.fit_uses_eval_set:
            fit_params.update({"eval_set": [(X_val, y_val)], "verbose": False})
        model.fit(X_train, y_train, **fit_params)
        return model

    def _process_shap_output(self, explanation: Any) -> tuple[np.ndarray, np.ndarray]:
        """处理 SHAP 输出, 确保兼容多分类."""
        shap_vals = explanation.values  # noqa: PD011
        base_vals = explanation.base_values
        if shap_vals.ndim == 3:  # 多分类任务
            shap_vals = np.abs(shap_vals).mean(axis=-1)
            base_vals = base_vals[:, 0] if base_vals.ndim > 1 else base_vals
        return shap_vals, base_vals

    def _calculate_stats(self) -> None:
        """计算 P-value 和得分 (Ratio)."""
        # P-value: (Null >= Real) / (N + 1). 使用修正项避免 p=0.
        count_exceed = np.sum(self.null_importances_distribution_ >= self.real_importances_, axis=0)
        self.p_values_ = (count_exceed + 1) / (len(self.null_importances_distribution_) + 1)

        # Ratio: Real / Mean(Null).
        mean_null = np.mean(self.null_importances_distribution_, axis=0)
        safe_mean_null = np.where(mean_null == 0, 1e-9, mean_null)
        self.scores_ = self.real_importances_ / safe_mean_null

        # 统一基类所需的重要性属性
        # 根据 mode 选择合适的指标作为"分数"，越高越好
        imp_values = 1.0 - self.p_values_ if self.mode == "p_value" else self.scores_

        self.feature_importances_ = pd.Series(imp_values, index=self.feature_names_in_)
