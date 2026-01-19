from typing import Any, Literal, cast

import pandas as pd

from ..base import Selector


class SelectNARatio(Selector):
    """缺失值过滤器 (基于 TransformerMixin 实现).

    该过滤器在 fit 阶段计算每个特征的缺失值比例/数量，
    并在 transform 阶段移除超过阈值的特征。

    Attributes:
        na_ratios_ (pd.Series): 拟合时计算的特征缺失值比例
        na_counts_ (pd.Series): 拟合时计算的特征缺失值数量
        n_features_in_ (int): 输入特征数量
        feature_names_in_ (np.ndarray): 输入特征名称 (如果是 DataFrame)
    """

    na_ratios_: pd.Series
    na_counts_: pd.Series

    def __init__(
        self,
        na_threshold: float = 0,
        strategy: Literal["ratio", "count"] = "ratio",
        **kwargs,
    ):
        """缺失值过滤器.

        Args:
            na_threshold: 缺失值阈值. 缺失值比例或数量**高于**此阈值的特征将被移除。
            strategy: 缺失值策略，默认"ratio"。
                - "ratio": 缺失值比例阈值，范围[0, 1]。
                - "count": 缺失值数量阈值，范围[0, inf]。
            **kwargs: 传递给父类的参数.
        """
        super().__init__(**kwargs)
        self.na_threshold = na_threshold
        self.strategy = strategy

        if strategy == "ratio" and not (0 <= na_threshold <= 1):
            raise ValueError("当strategy='ratio'时，threshold必须在[0, 1]范围内")
        if strategy == "count" and na_threshold < 0:
            raise ValueError("当strategy='count'时，threshold必须 >= 0")

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "SelectNARatio":
        """拟合选择器.

        计算 X 中各列的缺失值统计信息。

        Args:
            X: 输入特征 DataFrame
            y: 目标变量 (未使用)
            **kwargs: 额外参数.
        """
        # 1. 计算统计量
        self.na_counts_ = cast(pd.Series, X.isna().sum(axis=0))
        self.na_ratios_ = cast(pd.Series, X.isna().mean(axis=0))

        # 2. 计算需要移除的特征
        remove_mask = (
            self.na_ratios_ > self.na_threshold if self.strategy == "ratio" else self.na_counts_ > self.na_threshold
        )

        # 3. 记录保留的特征
        self.selected_features_ = X.columns[~remove_mask].tolist()

        return self
