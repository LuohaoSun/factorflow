from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import (
    chi2 as sk_chi2,
    mutual_info_classif,
    mutual_info_regression,
)

from ..base import Selector


class BaseSelectPercentile(Selector):
    """支持 NaN 处理的 SelectPercentile 基类.

    筛选 Score 最高的 top percentile% 特征。

    Attributes:
        scores_ (np.ndarray): 特征评分。
    """

    scores_: np.ndarray

    def __init__(self, percentile: int = 10, **kwargs):
        """初始化.

        Args:
            percentile: 要保留的特征百分比 (0-100)。默认 10。
            **kwargs: 传递给父类的参数.
        """
        super().__init__(**kwargs)
        self.percentile = percentile

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "BaseSelectPercentile":
        """拟合选择器.

        Args:
            X: 输入特征 DataFrame.
            y: 目标变量.
            **kwargs: 额外参数.
        """
        # 1. 统一转为 numpy (允许 NaN)
        X_arr = X.to_numpy()
        y_arr = np.array(y)

        if y_arr.ndim > 1:
            y_arr = y_arr.ravel()

        n_features = X_arr.shape[1]
        self.scores_ = np.zeros(n_features)

        # 2. 核心循环：逐列计算统计量
        for i in range(n_features):
            col_data = X_arr[:, i]
            mask = ~pd.isna(col_data)

            if np.sum(mask) < 2:
                self.scores_[i] = -np.inf
                continue

            X_valid = col_data[mask]
            y_valid = y_arr[mask]

            try:
                score = self._compute_score(X_valid, y_valid)
                self.scores_[i] = score
            except Exception:
                self.scores_[i] = -np.inf

        # 处理可能的 NaN score
        self.scores_ = np.nan_to_num(self.scores_, nan=-np.inf)

        # 3. 选择 Top Percentile
        if self.percentile >= 100:
            mask_selected = np.ones(n_features, dtype=bool)
        elif self.percentile <= 0:
            mask_selected = np.zeros(n_features, dtype=bool)
        else:
            # 计算阈值: 比如保留 10%，即需要分数 >= 90% 分位数
            threshold = np.percentile(self.scores_, 100 - self.percentile)
            mask_selected = self.scores_ >= threshold
            # 排除分数极低（无效）的情况
            mask_selected = mask_selected & (self.scores_ > -np.inf)

        self.selected_features_ = X.columns[mask_selected].tolist()

        return self

    @abstractmethod
    def _compute_score(self, x_col: np.ndarray, y: np.ndarray) -> float:
        """子类实现：返回 score (越大越好)."""
        pass


class SelectPercentileMutualInfo(BaseSelectPercentile):
    """基于互信息的 Top Percentile 选择器."""

    def __init__(self, percentile: int = 10, task_type: str = "classification", **kwargs):
        """初始化.

        Args:
            percentile: 保留百分比
            task_type: 'classification' or 'regression'
            **kwargs: 额外参数
        """
        super().__init__(percentile=percentile, **kwargs)
        self.task_type = task_type
        base_params = {"protected_features_patterns", "selection_check", "check_features_patterns", "label"}
        self.mi_kwargs = {k: v for k, v in kwargs.items() if k not in base_params}

    def _compute_score(self, x_col: np.ndarray, y: np.ndarray) -> float:
        X_reshaped = x_col.reshape(-1, 1)
        if self.task_type == "classification":
            scores = mutual_info_classif(X_reshaped, y, **self.mi_kwargs)
        else:
            scores = mutual_info_regression(X_reshaped, y, **self.mi_kwargs)
        return float(scores[0])


class SelectPercentileAnova(BaseSelectPercentile):
    """基于 ANOVA F-value 的 Top Percentile 选择器 (分类目标)."""

    def _compute_score(self, x_col: np.ndarray, y: np.ndarray) -> float:
        classes = np.unique(y)
        if len(classes) < 2:
            return 0.0
        groups = [x_col[y == c] for c in classes]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return 0.0
        f_score, _ = stats.f_oneway(*groups)
        return float(f_score)


class SelectPercentileChi2(BaseSelectPercentile):
    """基于 Chi2 的 Top Percentile 选择器 (分类目标, 非负特征)."""

    def _compute_score(self, x_col: np.ndarray, y: np.ndarray) -> float:
        X_reshaped = x_col.reshape(-1, 1)
        score_arr, _ = sk_chi2(X_reshaped, y)
        return float(score_arr[0])
