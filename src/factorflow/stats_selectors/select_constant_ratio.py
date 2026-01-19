from typing import Any, Literal

import pandas as pd

from ..base import Selector


class SelectConstantRatio(Selector):
    """常量特征过滤器.

    该过滤器识别并移除以下类型的特征：
    1. 常量特征：所有值都相同（排除缺失值后）
    2. 准常量特征：某个值出现频率过高

    Attributes:
        unique_counts_ (pd.Series): 拟合时计算的特征唯一值数量
        max_frequency_ratios_ (pd.Series): 拟合时计算的特征最大值频率比例
    """

    unique_counts_: pd.Series
    max_frequency_ratios_: pd.Series

    def __init__(
        self,
        threshold: float = 0.95,
        strategy: Literal["constant", "quasi_constant", "both"] = "both",
        min_unique_values: int = 2,
        **kwargs,
    ):
        """常量特征过滤器.

        Args:
            threshold: 准常量阈值. 最大频率比例**高于**此阈值的特征将被移除。
            strategy: 策略. 保留常量特征或准常量特征或两者。
            min_unique_values: 最小唯一值数量. 唯一值数量**小于**此阈值的特征将被移除。
            **kwargs: 传递给父类的参数.
        """
        super().__init__(**kwargs)
        if not (0 < threshold <= 1):
            raise ValueError("threshold必须在(0, 1]范围内")
        if min_unique_values < 1:
            raise ValueError("min_unique_values必须 >= 1")

        self.threshold = threshold
        self.strategy = strategy
        self.min_unique_values = min_unique_values

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "SelectConstantRatio":
        """拟合选择器.

        计算 X 中各列的统计信息（唯一值数量、最大频率占比）。

        Args:
            X: 输入特征 DataFrame
            y: 目标变量 (未使用)
            **kwargs: 额外参数.
        """
        unique_counts = {}
        max_frequency_ratios = {}

        # 1. 迭代计算统计量
        for col in X.columns:
            series = X[col]
            try:
                vc = series.value_counts(dropna=True, normalize=True)
                n_unique = len(vc)

                if n_unique == 0:
                    # 全是 NaN 的情况
                    unique_counts[col] = 0
                    max_frequency_ratios[col] = 1.0  # 视为最坏的情况
                else:
                    unique_counts[col] = n_unique
                    max_frequency_ratios[col] = vc.iloc[0]  # 占比最大的值的比例
            except TypeError:
                # 处理不可哈希的类型
                unique_counts[col] = len(series.unique())
                max_frequency_ratios[col] = 0.0

        self.unique_counts_ = pd.Series(unique_counts)
        self.max_frequency_ratios_ = pd.Series(max_frequency_ratios)

        # 2. 计算 mask
        mask_constant = self.unique_counts_ >= self.min_unique_values
        mask_quasi = self.max_frequency_ratios_ <= self.threshold

        if self.strategy == "constant":
            mask = mask_constant
        elif self.strategy == "quasi_constant":
            mask = mask_quasi
        else:  # both
            mask = mask_constant & mask_quasi

        # 3. 记录保留的特征
        self.selected_features_ = X.columns[mask].tolist()

        return self
