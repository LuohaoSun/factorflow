from abc import abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import chi2 as sk_chi2

from ..base import Selector


class BaseSelectFpr(Selector):
    """支持 NaN 处理的 SelectFpr 基类.

    Args:
        alpha: 显著性水平阈值，默认 0.05。即允许的假阳性率上限。
        bonferroni_correction: 是否应用 Bonferroni 校正。

    Attributes:
        pvalues_ (np.ndarray): P 值。
        scores_ (np.ndarray): 特征评分。
    """

    pvalues_: np.ndarray
    scores_: np.ndarray

    def __init__(self, alpha: float = 0.05, bonferroni_correction: bool = False, **kwargs):
        """初始化.

        Args:
            alpha: 显著性水平阈值.
            bonferroni_correction: 是否应用 Bonferroni 校正.
            **kwargs: 传递给父类的参数.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.bonferroni_correction = bonferroni_correction

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "BaseSelectFpr":
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
        self.pvalues_ = np.ones(n_features)  # 默认 P=1 (不显著)
        self.scores_ = np.zeros(n_features)

        # 2. 核心循环：逐列计算统计量
        for i in range(n_features):
            col_data = X_arr[:, i]
            mask = ~pd.isna(col_data)

            if np.sum(mask) < 2:
                continue

            X_valid = col_data[mask]
            y_valid = y_arr[mask]

            try:
                score, pval = self._compute_statistic(X_valid, y_valid)
                self.scores_[i] = score
                self.pvalues_[i] = pval
            except Exception:
                # 发生错误，默认不显著
                self.scores_[i] = 0.0
                self.pvalues_[i] = 1.0

        # 处理可能的 NaN pvalue
        self.pvalues_ = np.nan_to_num(self.pvalues_, nan=1.0)

        # 3. 计算支持的特征
        threshold = self.alpha
        if self.bonferroni_correction:
            threshold = self.alpha / n_features

        mask_selected = self.pvalues_ < threshold
        self.selected_features_ = X.columns[mask_selected].tolist()

        return self

    @abstractmethod
    def _compute_statistic(self, x_col: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """子类必须实现此方法，返回 (score, pvalue)."""
        pass


class SelectFprAnova(BaseSelectFpr):
    """适用于：特征是连续数值，目标是分类标签.

    方法：One-way ANOVA (f_oneway)。
    """

    def _compute_statistic(self, x_col: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        classes = np.unique(y)
        if len(classes) < 2:
            return 0.0, 1.0

        groups = [x_col[y == c] for c in classes]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return 0.0, 1.0

        return stats.f_oneway(*groups)  # type: ignore


class SelectFprPearson(BaseSelectFpr):
    """适用于：特征是连续数值，目标是连续数值.

    方法：Pearson Correlation (转为 F-test).
    """

    def _compute_statistic(self, x_col: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        result = stats.pearsonr(x_col, y)
        r = float(cast(Any, result[0]))
        p = float(cast(Any, result[1]))
        return float(abs(r)), p


class SelectFprChi2(BaseSelectFpr):
    """适用于：特征是分类（离散整数），目标是分类标签.

    方法：Chi-Square Test.
    """

    def _compute_statistic(self, x_col: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        X_reshaped = x_col.reshape(-1, 1)
        score_arr, pval_arr = sk_chi2(X_reshaped, y)
        return float(score_arr[0]), float(pval_arr[0])


class SelectFprAnovaReverse(BaseSelectFpr):
    """适用于：特征是分类（作为分组变量），目标是连续数值.

    方法：反向 ANOVA (以 X 分组，看 Y 的均值是否有显著差异).
    """

    def _compute_statistic(self, x_col: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        groups_keys = np.unique(x_col)
        if len(groups_keys) < 2:
            return 0.0, 1.0

        groups_values = [y[x_col == k] for k in groups_keys]
        groups_values = [g for g in groups_values if len(g) > 0]

        if len(groups_values) < 2:
            return 0.0, 1.0

        return stats.f_oneway(*groups_values)  # type: ignore


class SelectFprKS(BaseSelectFpr):
    """适用于：特征是连续数值，目标是二分类标签.

    方法：Kolmogorov-Smirnov Test (ks_2samp).
    """

    def _compute_statistic(self, x_col: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        classes = np.unique(y)
        if len(classes) != 2:
            return 0.0, 1.0

        group1 = x_col[y == classes[0]]
        group2 = x_col[y == classes[1]]

        if len(group1) == 0 or len(group2) == 0:
            return 0.0, 1.0

        res = stats.ks_2samp(group1, group2)
        return float(res.statistic), float(res.pvalue)  # type: ignore
