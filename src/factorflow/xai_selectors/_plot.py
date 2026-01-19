from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    import mlflow
except ImportError:
    mlflow = None


def plot_oof_shap_summary(
    selector: any,
    X: pd.DataFrame,
    max_display: int = 20,
):
    """绘制 Out-of-Fold (OOF) SHAP 摘要图并可选地记录到 MLflow.

    该函数会生成 Beeswarm 图（展示特征对预测的影响分布）和 Bar 图（展示平均绝对影响力）。
    如果是活跃的 MLflow 运行，图像将作为 artifacts 存储在 `model/plots/{label}/` 目录下。

    Args:
        selector: 训练好的 SHAP 选择器实例（需包含 shap_values_oof_ 等属性）.
        X: 原始特征 DataFrame, 用于提供特征名称和对齐数据.
        max_display: 图中显示的最大特征数量.
    """
    valid_mask = ~np.isnan(selector.shap_values_oof_).any(axis=1)
    if np.sum(valid_mask) <= 0:
        logger.warning("No valid SHAP values found for plotting.")
        return

    try:
        data_source = selector.shap_data_oof_ if selector.shap_data_oof_ is not None else X

        # 构造 shap.Explanation 对象，它是 SHAP 绘图新版 API 的标准输入
        final_explanation = shap.Explanation(
            values=selector.shap_values_oof_[valid_mask],
            data=data_source.iloc[valid_mask].to_numpy(),
            feature_names=data_source.columns.tolist(),
            base_values=selector.base_values_oof_[valid_mask],
        )

        label = selector.label or selector.__class__.__name__
        artifact_dir = f"model/plots/{label}"

        if mlflow and mlflow.active_run():
            # 根据特征名长度动态计算图宽度
            max_name_len = max(len(str(name)) for name in data_source.columns)
            # 左边距比例：根据最长特征名计算，每个字符约占 0.008 的比例
            left_margin = min(0.5, 0.05 + max_name_len * 0.008)
            fig_width = max(12, 8 + max_name_len * 0.15)
            fig_height = max(8, max_display * 0.4)

            # 1. Beeswarm Plot
            fig = plt.figure(figsize=(fig_width, fig_height))
            shap.plots.beeswarm(final_explanation, show=False, max_display=max_display)
            plt.title(f"SHAP Beeswarm ({selector.n_splits}-Fold CV)")
            plt.subplots_adjust(left=left_margin, right=0.95, top=0.95, bottom=0.08)
            mlflow.log_figure(fig, f"{artifact_dir}/shap_beeswarm_cv.png")
            plt.close(fig)

            # 2. Bar Plot
            fig = plt.figure(figsize=(fig_width, fig_height))
            shap.plots.bar(final_explanation, show=False, max_display=max_display)
            plt.title(f"SHAP Bar ({selector.n_splits}-Fold CV)")
            plt.subplots_adjust(left=left_margin, right=0.95, top=0.95, bottom=0.08)
            mlflow.log_figure(fig, f"{artifact_dir}/shap_bar_cv.png")
            plt.close(fig)

            logger.info(f"SHAP plots recorded to MLflow artifact path: {artifact_dir}")
        else:
            # 本地交互式显示（回退到旧版 API 的 summary_plot 以获得更好的兼容性）
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                selector.shap_values_oof_[valid_mask], data_source.iloc[valid_mask], max_display=max_display, show=True
            )

    except Exception as e:
        logger.warning(f"Failed to create SHAP summary plot: {e}")


def plot_null_importance_distributions(
    selector: any,
    top_k: int = 10,
):
    """绘制 Top 特征的真实重要性与 Null 重要性分布的对比图.

    该函数会为每一个 Top 特征生成一个子图，展示 Target Permutation 产生的零分布（灰色），
    并标记出真实数据的特征重要性（红色虚线）。这有助于直观判断特征是否显著。

    Args:
        selector: 训练好的 NullImportance 选择器实例.
        top_k: 要绘制分布对比的特征数量.
    """
    import seaborn as sns

    # 按照 Base Run 的真实重要性降序排列
    top_indices = np.argsort(selector.real_importances_)[::-1][:top_k]
    n_features = len(top_indices)

    if n_features == 0:
        logger.warning("No features to plot null distribution.")
        return

    # 动态计算网格布局
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, idx in enumerate(top_indices):
        ax = axes[i]
        feat_name = selector.feature_names_in_[idx]
        real_val = selector.real_importances_[idx]
        null_vals = selector.null_importances_distribution_[:, idx]

        # 绘制零分布（置换后产生的噪音重要性）
        sns.histplot(null_vals, color="gray", label="Null Distribution", kde=True, ax=ax)
        # 标记真实值
        ax.axvline(x=float(real_val), color="red", linestyle="--", linewidth=2, label=f"Real Imp ({real_val:.4f})")

        p_val = selector.p_values_[idx]
        ratio = selector.scores_[idx]
        ax.set_title(f"{feat_name}\nP-val: {p_val:.4f}, Ratio: {ratio:.2f}")
        ax.legend()

    # 隐藏多余的空白坐标轴
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    label = selector.label or selector.__class__.__name__
    if mlflow and mlflow.active_run():
        artifact_path = f"model/plots/{label}/null_importance_dist.png"
        mlflow.log_figure(fig, artifact_path)
        logger.info(f"Null importance distributions recorded to MLflow: {artifact_path}")
    else:
        plt.show()

    plt.close()
