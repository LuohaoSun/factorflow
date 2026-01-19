from typing import Any, cast

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..base import Selector


def _calc_single_vif(col: str, X: pd.DataFrame) -> float:
    """计算单个特征的 VIF 值 (回归法).

    Args:
        col: 目标特征名.
        X: 包含所有特征的 DataFrame.

    Returns:
        VIF 值.
    """
    y_target = X[col]
    X_other = X.drop(columns=[col])

    # 如果没有其他特征，VIF = 1.0
    if X_other.shape[1] == 0:
        return 1.0

    lr = LinearRegression()
    lr.fit(X_other, y_target)
    r2 = lr.score(X_other, y_target)

    # 处理完美的共线性
    if r2 >= 0.99999:
        return float("inf")
    return 1.0 / (1.0 - r2)


class SelectVif(Selector):
    """VIF (Variance Inflation Factor) 特征选择器.

    通过迭代计算 VIF 并移除最高 VIF 的特征，直到所有特征的 VIF 都低于阈值。
    兼容 NaN (计算时使用中位数填充，如果整列 NaN 则填 0).

    Attributes:
        vifs_ (pd.Series): 拟合后最终保留特征的 VIF 值。
    """

    vifs_: pd.Series

    def __init__(self, threshold: float = 5.0, step: int | float = 1, n_jobs: int = -1, **kwargs):
        """初始化.

        Args:
            threshold: VIF 阈值，用于判断特征间多重共线性的严重程度。
                - 较低的阈值（例如 2~5）：适合需要严格消除多重共线性的场景（如回归分析的解释性更重要时）。
                - 阈值为 5（常用默认）：适度控制多重共线性，兼顾解释性和特征利用。
                - 较高的阈值（如 10 及以上）：只剔除极其严重共线的特征。
            step: 每次迭代移除的特征数量。
                - 如果是整数 >= 1，表示移除 N 个特征.
                - 如果是浮点数 0 < step < 1，表示移除剩余特征的比例 (例如 0.1 = 10%).
            n_jobs: 并行计算的作业数（仅在回退到回归法时使用）。
            **kwargs: 传递给父类的参数.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.step = step
        self.n_jobs = n_jobs

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "SelectVif":
        """拟合选择器.

        Args:
            X: 输入特征 DataFrame.
            y: 目标变量.
            **kwargs: 额外参数.
        """
        # 1. 筛选数值列 (VIF 仅对数值特征有意义)
        X_numeric = X.select_dtypes(include=[np.number])
        non_numeric_cols = [c for c in X.columns if c not in X_numeric.columns]

        # 2. 处理 NaN (仅用于计算 VIF，不修改原始数据)
        # 使用中位数填充，比均值更抗干扰; 全列 NaN 填 0
        X_filled = X_numeric.fillna(X_numeric.median()).fillna(0.0)

        features = list(X_numeric.columns)

        # 3. 迭代筛选
        while features:
            X_curr = cast(pd.DataFrame, X_filled[features])
            n_feats = len(features)

            # --- 尝试矩阵加速法 (Matrix Method) ---
            vifs: list[float]
            try:
                # VIF 等于相关系数矩阵逆矩阵的对角线元素
                corr_matrix = X_curr.corr()
                inv_corr = np.linalg.inv(corr_matrix.values)
                vifs_arr = np.diag(inv_corr)
                # 处理数值误差导致的微小负值
                vifs = [max(1.0, v) for v in vifs_arr]
            except np.linalg.LinAlgError:
                # --- 回退到回归法 (Regression Method) ---
                # 矩阵不可逆通常意味着存在完美共线性 (VIF=inf)
                if n_feats < 10:
                    vifs = [_calc_single_vif(f, X_curr) for f in features]
                else:
                    results = Parallel(n_jobs=self.n_jobs)(delayed(_calc_single_vif)(f, X_curr) for f in features)
                    vifs = list(results)  # type: ignore

            # 检查最大 VIF
            max_vif = max(vifs) if vifs else 0.0
            if max_vif <= self.threshold:
                self.vifs_ = pd.Series(vifs, index=features)
                break

            # --- 确定要移除的特征 ---
            # 创建 (vif, feature_name) 的列表并排序
            vif_pairs = sorted(zip(vifs, features, strict=True), key=lambda x: x[0], reverse=True)

            # 计算本轮要移除的数量
            n_to_remove = 1
            if isinstance(self.step, int) and self.step > 1:
                n_to_remove = self.step
            elif isinstance(self.step, float) and 0 < self.step < 1:
                n_to_remove = max(1, int(n_feats * self.step))

            # 只有 VIF > threshold 的才会被移除
            remove_candidates = [(v, f) for v, f in vif_pairs if v > self.threshold]

            # 真正移除的数量取两者较小值
            real_remove_count = min(len(remove_candidates), n_to_remove)

            # 执行移除
            for i in range(real_remove_count):
                feat_to_remove = remove_candidates[i][1]
                features.remove(feat_to_remove)

        # 4. 整理结果
        # 保留的特征 = 筛选剩下的数值特征 + 非数值特征
        self.selected_features_ = features + non_numeric_cols

        return self
