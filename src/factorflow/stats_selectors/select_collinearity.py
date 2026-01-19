from typing import Any, Literal

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ..base import Selector


class SelectCollinearity(Selector):
    """基于共线性的特征选择器.

    自动识别高度相关的特征组，并从每组中保留一个代表性特征。

    Attributes:
        _internal_selector (Any): 内部使用的 feature_engine.selection.SmartCorrelatedSelection 实例。
    """

    _internal_selector: Any

    def __init__(
        self,
        threshold: float = 0.95,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        keep_strategy: Literal["least_na", "variance", "cardinality"] = "least_na",
        missing_values: Literal["raise", "ignore"] = "ignore",
        estimator: Any = None,
        **kwargs,
    ):
        """基于共线性的特征选择器.

        Args:
            threshold: 相关系数阈值 (0到1)。默认 0.95。
            method: 相关系数计算方法。
            keep_strategy: 保留策略。映射到 feature-engine 的 selection_method。
                - "least_na": 保留缺失值最少的。
                - "variance": 保留方差最大的。
                - "cardinality": 保留唯一值数量最多的。
                - "model_performance": (需传入 estimator) 保留模型重要性最高的。
            missing_values: 计算相关性时如何处理缺失值 ("ignore" 会自动忽略 NA 行)。
            estimator: 当 keep_strategy="model_performance" 时使用的基模型。
            **kwargs: 传递给父类的参数.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.method = method
        self.keep_strategy = keep_strategy
        self.missing_values = missing_values
        self.estimator = estimator

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "SelectCollinearity":
        """拟合选择器.

        Args:
            X: 输入特征 DataFrame.
            y: 目标变量.
            **kwargs: 额外参数.
        """
        # Lazy import
        from feature_engine.selection import SmartCorrelatedSelection

        # 映射 keep_strategy 到 feature_engine 的参数名
        strategy_map = {
            "least_na": "missing_values",
            "variance": "variance",
            "cardinality": "cardinality",
        }
        selection_method = strategy_map.get(self.keep_strategy, self.keep_strategy)

        # 初始化内部 selector
        self._internal_selector = SmartCorrelatedSelection(
            variables=None,  # 自动检测数值变量
            method=self.method,
            threshold=self.threshold,
            missing_values=self.missing_values,
            selection_method=selection_method,
            estimator=self.estimator,
            scoring="roc_auc" if self.estimator else None,
            cv=3,
        )

        # 委托 fit
        self._internal_selector.fit(X, y)

        # 记录选中的特征
        # feature_names_in_ 已经在 base.fit 中设置
        drop_set = set(self._internal_selector.features_to_drop_)
        self.selected_features_ = [f for f in self.feature_names_in_ if f not in drop_set]

        return self

    def get_feature_groups(self) -> dict[str, list[str]]:
        """获取特征分组详情.

        Returns:
            dict[str, list[str]]: 字典形式: { '保留的特征名': ['被移除的特征A', '被移除的特征B'] }
            如果没有特征被移除，返回空字典。
        """
        check_is_fitted(self, ["_internal_selector"])

        groups = {}
        features_to_drop_ = self._internal_selector.features_to_drop_
        correlated_feature_sets_ = self._internal_selector.correlated_feature_sets_

        drop_set = set(features_to_drop_)

        for group_set in correlated_feature_sets_:
            kept_features = group_set - drop_set
            kept_in_group = list(kept_features)

            if not kept_in_group:
                continue

            kept_feat = kept_in_group[0]
            dropped_members = list(group_set & drop_set)

            if dropped_members:
                groups[kept_feat] = dropped_members

        return groups
