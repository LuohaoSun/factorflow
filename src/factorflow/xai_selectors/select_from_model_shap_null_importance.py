from typing import Any, Literal, cast

from loguru import logger
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

from factorflow.base import Selector

from ._plot import plot_null_importance_distributions
from .select_from_model_shap_cv import SelectFromModelShapCV


class SelectFromModelShapNullImportance(Selector):
    """基于 Null Importance (目标变量置换) 的 SHAP 特征选择器 (V2).

    算法原理:
    1. Base Run: 在原始数据上使用交叉验证计算稳健的 SHAP 特征重要性（真实重要性）。
    2. Null Runs: 多次随机打乱目标变量 y 的顺序（Target Permutation），并重新训练模型计算 SHAP 重要性。
       这一步构建了每个特征在“无意义数据”上的重要性零分布 (Null Distribution)。
    3. 显著性评估:
       - P-value: 计算零分布中重要性大于或等于真实重要性的次数占比。若 P-value 较小，说明该特征的重要性
         并非偶然获得。
       - Ratio: 计算真实重要性与零分布均值的比率。若 Ratio 远大于 1，说明该特征贡献显著。

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
        """初始化 Null Importance 选择器.

        Args:
            estimator: 基础估计器 (需符合 sklearn API).
            task_type: 任务类型, "classification" 或 "regression".
            n_trials: 目标变量置换的次数 (实验次数). 建议至少 50 次以获得稳健的分布.
            mode: 筛选模式.
                - "p_value": 计算 P 值 (Null >= Real 的占比), 保留 p < threshold 的特征.
                - "ratio": 计算 Real / Mean(Null), 保留比率 > threshold 的特征.
            threshold: 筛选阈值. p_value 模式下通常设为 0.05; ratio 模式下通常设为 1.0 以上.
            val_size: Null Runs 阶段 Hold-out 验证集的比例.
            random_state: 随机种子, 确保实验可重复.
            shap_sample_size: 计算 SHAP 值时的样本采样量, 设为 None 则使用全量验证集.
            fit_uses_eval_set: 模型训练时是否需要传入 eval_set (如梯度提升树的早停).
            verbose: 日志详尽程度. 0: 静默; 1: 打印指标; 2: 打印指标并本地绘图.
            max_display: 报告和绘图时显示的 Top 特征数量.
            model_fit_params: 透传给 estimator.fit 的额外参数.
            cv: Base Run 阶段用于计算真实重要性的交叉验证折数.
            **kwargs: 透传给父类 BaseSelector 的参数 (如 label, protected_features_patterns 等).
        """
        super().__init__(**kwargs)
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
        """执行特征选择逻辑.

        包含两个阶段:
        1. Base Run: 使用 SelectFromModelShapCV 获取原始数据上的稳健 SHAP 重要性.
        2. Null Runs: 多次执行 Target Permutation 训练并收集零分布.
        """
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
        self.real_importances_ = base_selector.feature_importances_

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
            explanation = explainer(X_shap_val)

            shap_vals, _ = self._process_shap_output(explanation)
            null_imp = np.abs(shap_vals).mean(axis=0)
            null_imps_list.append(null_imp)

        self.null_importances_distribution_ = np.array(null_imps_list)

        # 3. 统计指标计算
        self._calculate_stats()
        self.selected_features_ = self._get_selected_features()

        # 4. 报告与可视化
        if self.verbose:
            self._log_summary()

        if self.verbose >= 2 or (self.verbose >= 1 and mlflow and mlflow.active_run()):
            plot_null_importance_distributions(self, top_k=self.max_display)

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
        self.feature_importances_ = self.real_importances_

    def _log_summary(self, k: int = 10) -> None:
        """打印显著性分析摘要日志."""
        df = pd.DataFrame(
            {
                "feature": self.feature_names_in_,
                "real_imp": self.real_importances_,
                "p_value": self.p_values_,
                "ratio": self.scores_,
            }
        )
        top_df = df.sort_values(["p_value", "real_imp"], ascending=[True, False]).head(k)
        logger.info(f"[{self.label}] Null Importance Summary (Top {k}):\n{top_df.to_string(index=False)}")

        if mlflow and mlflow.active_run():
            mlflow.log_metric(f"{self.label}_selected_count", len(self.selected_features_))
