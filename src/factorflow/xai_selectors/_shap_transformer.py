from typing import Any, Literal

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted


class ShapTransformer(BaseEstimator, TransformerMixin):
    """训练一个模型, 将 X 转换到 SHAP 空间.

    该转换器旨在将输入特征 X 映射到其对应的 SHAP 值空间。
    它使用提供的估计器在训练数据上进行拟合，然后利用 SHAP 解释器计算特征的贡献。

    Attributes
    ----------
        estimator_ (BaseEstimator): 拟合后的基础估计器副本。
        explainer_ (shap.Explainer): 用于计算 SHAP 值的解释器。
        feature_names_in_ (np.ndarray): 输入特征名称。
        feature_importances_ (pd.Series): 基于训练数据 SHAP 绝对值均值的特征重要性。
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        task_type: Literal["classification", "regression"] = "regression",
        enable_multiclass: bool = False,
    ):
        """初始化 ShapTransformer.

        Args:
        ----
            estimator: 用于计算 SHAP 值的 sklearn 兼容估计器。
            task_type: 任务类型, "classification" 或 "regression"。
            enable_multiclass: 是否启用多分类支持。默认为 False。
        """
        self.estimator = estimator
        self.task_type = task_type
        self.enable_multiclass = enable_multiclass

    def fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "ShapTransformer":
        """拟合模型并初始化 SHAP 解释器.

        Args:
        ----
            X: 输入特征 DataFrame。
            y: 目标变量。
            **kwargs: 透传给 estimator.fit 的额外参数。

        Returns:
        -------
            self
        """
        self.feature_names_in_ = np.array(X.columns.tolist())

        # 克隆并拟合模型
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]

        # 初始化解释器
        # 使用 shap.Explainer 自动选择最佳解释器 (TreeExplainer, LinearExplainer 等)
        self.explainer_ = shap.Explainer(self.estimator_, X)

        # 计算特征重要性 (基于拟合时使用的训练数据 X)
        explanation = self.explainer_(X, check_additivity=False)
        shap_values = explanation.values  # noqa: PD011

        # 维度转换逻辑: 确保 shap_values 最终为 (n_samples, n_features)
        if shap_values.ndim == 3 and shap_values.shape[2] == 2:
            # 二分类情况: 取正类贡献
            shap_values = shap_values[:, :, 1]

        if shap_values.ndim == 3:
            # 真正的多分类情况 (> 2 classes)
            if not self.enable_multiclass:
                raise ValueError(
                    "Detected multiclass SHAP values (ndim=3, n_classes > 2), but multiclass support is disabled. "
                    "To enable it, set `enable_multiclass=True` during initialization."
                )

            import warnings

            warnings.warn(
                "Multiclass support in ShapTransformer is experimental. "
                "Collapsing class dimension by taking mean of absolute SHAP values.",
                UserWarning,
                stacklevel=2,
            )
            shap_values = np.abs(shap_values).mean(axis=-1)

        importances_arr = np.abs(shap_values).mean(axis=0)
        self.feature_importances_ = pd.Series(importances_arr, index=self.feature_names_in_)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """将 X 转换为 SHAP 值空间.

        Args:
        ----
            X: 输入特征 DataFrame。

        Returns:
        -------
            pd.DataFrame: 包含 SHAP 值的 DataFrame，索引和列名与输入 X 保持一致。
        """
        check_is_fitted(self, ["estimator_", "explainer_", "feature_names_in_", "feature_importances_"])

        # 计算 SHAP 值
        explanation = self.explainer_(X, check_additivity=False)
        shap_values = explanation.values  # noqa: PD011

        # 维度转换逻辑 (与 fit 保持一致)
        if shap_values.ndim == 3 and shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]

        if shap_values.ndim == 3:
            shap_values = np.abs(shap_values).mean(axis=-1)

        return pd.DataFrame(shap_values, index=X.index, columns=X.columns)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """获取输出特征名称 (与输入一致)."""
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_
