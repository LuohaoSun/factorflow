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


class BaseSelectKBest(Selector):
    """支持 NaN 处理的 SelectKBest 基类.

    筛选 Score 最高的 k 个特征。

    Attributes:
        scores_ (np.ndarray): 特征评分。
    """

    scores_: np.ndarray

    def __init__(self, k: int = 10, **kwargs):
        """初始化.

        Args:
            k: 要选择的特征数量。
            **kwargs: 传递给父类的参数.
        """
        super().__init__(**kwargs)
        self.k = k

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "BaseSelectKBest":
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

        # 3. 选择 Top K
        k_select = min(self.k, n_features)

        if k_select == 0:
            self.selected_features_ = []
        else:
            # argsort 是升序，取最后 k 个
            top_k_indices = np.argsort(self.scores_)[-k_select:]
            self.selected_features_ = X.columns[top_k_indices].tolist()

        return self

    @abstractmethod
    def _compute_score(self, x_col: np.ndarray, y: np.ndarray) -> float:
        """子类实现：返回 score (越大越好)."""
        pass


class SelectKBestMutualInfo(BaseSelectKBest):
    """基于互信息的 Top K 选择器."""

    def __init__(self, k: int = 10, task_type: str = "classification", **kwargs):
        """初始化.

        Args:
            k: 选择特征数
            task_type: 'classification' or 'regression'
            **kwargs: 传递给 mutual_info_classif 或 mutual_info_regression 的参数
        """
        super().__init__(k=k, **kwargs)
        self.task_type = task_type
        # 保存 kwargs 以便在 _compute_score 中使用，
        # 排除 BaseSelector 的参数
        base_params = {"protected_features_patterns", "selection_check", "check_features_patterns", "label"}
        self.mi_kwargs = {k: v for k, v in kwargs.items() if k not in base_params}

    def _compute_score(self, x_col: np.ndarray, y: np.ndarray) -> float:
        X_reshaped = x_col.reshape(-1, 1)

        if self.task_type == "classification":
            scores = mutual_info_classif(X_reshaped, y, **self.mi_kwargs)
        else:
            scores = mutual_info_regression(X_reshaped, y, **self.mi_kwargs)

        return float(scores[0])


class SelectKBestAnova(BaseSelectKBest):
    """基于 ANOVA F-value 的 Top K 选择器 (分类目标)."""

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


class SelectKBestChi2(BaseSelectKBest):
    """基于 Chi2 的 Top K 选择器 (分类目标, 非负特征)."""

    def _compute_score(self, x_col: np.ndarray, y: np.ndarray) -> float:
        X_reshaped = x_col.reshape(-1, 1)
        score_arr, _ = sk_chi2(X_reshaped, y)
        return float(score_arr[0])
