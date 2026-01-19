from itertools import combinations
import logging
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd
from scipy import stats

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore

logger = logging.getLogger(__name__)


def _calc_target_mode(
    feat: str, t_vals: np.ndarray, r_vals: np.ndarray, target_group: Any, ref_name: str
) -> dict[str, Any]:
    """Worker function for Target mode."""
    # Metric 1: Wasserstein Distance
    drift_score = stats.wasserstein_distance(t_vals, r_vals)
    # Metric 2: Mean Difference
    diff_mean = np.mean(t_vals) - np.mean(r_vals)
    # Metric 3: Mann-Whitney U
    try:
        _, p_val = stats.mannwhitneyu(t_vals, r_vals)
    except ValueError:
        p_val = 1.0

    return {
        "feature": feat,
        "drift_score": drift_score,
        "diff_mean": diff_mean,
        "p_value": p_val,
        f"mean_{target_group}": np.mean(t_vals),
        f"mean_{ref_name}": np.mean(r_vals),
    }


def _calc_multigroup_mode(
    feat: str, group_data_map: dict[Any, np.ndarray], unique_groups: np.ndarray
) -> dict[str, Any]:
    """Worker function for Multi-group mode."""
    group_means = [np.mean(vals) for vals in group_data_map.values()]

    # Metric 1: Max Pairwise Wasserstein
    ws_distances = []
    for g1, g2 in combinations(unique_groups, 2):
        d = stats.wasserstein_distance(group_data_map[g1], group_data_map[g2])
        ws_distances.append(d)
    drift_score = np.max(ws_distances) if ws_distances else 0.0

    # Metric 2: Max Mean Range
    diff_mean_range = np.max(group_means) - np.min(group_means)

    # Metric 3: Kruskal-Wallis ANOVA
    try:
        _, p_val = stats.kruskal(*group_data_map.values())
    except ValueError:
        p_val = 1.0

    res = {
        "feature": feat,
        "drift_score": drift_score,
        "diff_mean_range": diff_mean_range,
        "p_value_anova": p_val,
    }
    for g, m in zip(unique_groups, group_means, strict=False):
        res[f"mean_{g}"] = m
    return res


def grouped_shap_diff(
    shap_values: pd.DataFrame | np.ndarray,
    groups: pd.Series | np.ndarray | list[Any],
    target_group: Any | None = None,
    reference_group: Any | None = None,
    feature_names: list[str] | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """计算 SHAP 值在不同组别间的分布漂移 (Drift)。

    自动适配 "两组对比" 和 "多组波动分析" 两种场景。

    Args:
        shap_values: SHAP 矩阵 (DataFrame 或 ndarray)。
        groups: 分组标签。
        target_group: (可选) 指定关注的目标组。
            - 如果提供，进行 Target vs Reference (或 Rest) 的定向对比。
            - 如果不提供，进行所有组之间的两两对比，寻找最大差异。
        reference_group: (可选) 指定对照组。仅在 target_group 存在时有效。
        feature_names: (可选) 当输入为 ndarray 时，指定特征名称。
        n_jobs: (可选) 并行任务数，默认为 -1 (使用所有 CPU 核心)。如果 joblib 未安装，将回退到串行。

    Returns:
        pd.DataFrame: 包含 drift_score, diff_mean, p_value 等指标，按 drift_score 降序排列。

    Raises:
        ValueError: 当计算复杂度过高或组数过多时抛出。
    """
    # --- 1. 数据标准化 ---
    df: pd.DataFrame
    if isinstance(shap_values, np.ndarray):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
        df = pd.DataFrame(shap_values, columns=cast(Any, feature_names))
    else:
        df = shap_values.copy()

    groups = pd.Series(groups).reset_index(drop=True)
    df = df.reset_index(drop=True)
    unique_groups = groups.unique()
    n_features = len(df.columns)
    n_groups = len(unique_groups)

    # --- 2. 复杂度检测 ---
    # 规则 1: 多组全量两两对比模式下，组数不应过多
    if target_group is None and n_groups > 20:
        raise ValueError(
            f"Too many groups for pairwise comparison: {n_groups} groups found (limit: 20). "
            "Please reduce groups or specify a `target_group`."
        )

    # 规则 2: 估算计算量 (两两比较次数 = 特征数 * 组组合数)
    # C(N, 2) = N * (N-1) / 2
    if target_group is None:
        pairwise_comparisons = n_features * (n_groups * (n_groups - 1) // 2)
        # 设定一个硬性阈值，比如 20000 次比较
        # (例如 100 特征 * 20 组 = 100 * 190 = 19000，这是勉强可接受的上限)
        if pairwise_comparisons > 20000:
            raise ValueError(
                f"Computation too heavy: {pairwise_comparisons} pairwise comparisons required "
                f"(Limit: 20000). Please reduce features or groups."
            )

    # 检查并行环境
    use_parallel = False
    if n_jobs != 1:
        if Parallel is not None:
            use_parallel = True
        else:
            warnings.warn("joblib not installed, falling back to serial execution.", stacklevel=2)

    results = []

    # =======================================================
    # 场景 A: 明确指定了 Target (2组定向对比模式)
    # =======================================================
    if target_group is not None:
        target_data = df[groups == target_group]
        if reference_group is not None:
            ref_data = df[groups == reference_group]
            ref_name = str(reference_group)
        else:
            ref_data = df[groups != target_group]
            ref_name = "Rest"

        if target_data.empty or ref_data.empty:
            raise ValueError(f"组别样本不足: Target={len(target_data)}, Ref={len(ref_data)}")

        if use_parallel and Parallel and delayed:
            # 预取数据为 numpy array，避免在并行中传递 DataFrame
            # joblib 传递大对象会有序列化开销，但切分后再传开销较小
            tasks = [
                delayed(_calc_target_mode)(
                    feat,
                    np.array(target_data[feat]),
                    np.array(ref_data[feat]),
                    target_group,
                    ref_name,
                )
                for feat in df.columns
            ]
            results = Parallel(n_jobs=n_jobs)(tasks)
        else:
            for feat in df.columns:
                results.append(
                    _calc_target_mode(
                        feat,
                        np.array(target_data[feat]),
                        np.array(ref_data[feat]),
                        target_group,
                        ref_name,
                    )
                )

    # =======================================================
    # 场景 B: 没有指定 Target (多组全量扫描模式)
    # =======================================================
    else:
        # 预先将所有组的数据切分为 dict[str, dict[group, array]] 格式不太好传
        # 还是按 feature 循环切分比较好

        if use_parallel and Parallel and delayed:
            tasks = []
            for feat in df.columns:
                # 提取该特征下所有组的数据
                group_data_map = {g: np.array(df[groups == g][feat]) for g in unique_groups}
                tasks.append(delayed(_calc_multigroup_mode)(feat, group_data_map, np.array(unique_groups)))
            results = Parallel(n_jobs=n_jobs)(tasks)
        else:
            for feat in df.columns:
                group_data_map = {g: np.array(df[groups == g][feat]) for g in unique_groups}
                results.append(_calc_multigroup_mode(feat, group_data_map, np.array(unique_groups)))

    # 3. 统一输出
    # results 此时是 List[Dict] (无论是并行还是串行)
    return pd.DataFrame(results).set_index("feature").sort_values("drift_score", ascending=False)
