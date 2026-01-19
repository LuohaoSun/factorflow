from typing import TYPE_CHECKING, Any

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# 设置中文显示，解决 Glyph 缺失警告
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",  # macOS 常用
    "PingFang SC",  # macOS 常用
    "SimHei",  # Windows 常用
    "Noto Sans CJK SC",  # Linux (Ubuntu/Debian) 常用
    "WenQuanYi Micro Hei",  # Linux (CentOS/Fedora) 常用
    "Droid Sans Fallback",  # 旧版 Linux 常用
    "DejaVu Sans",  # 兜底
]
plt.rcParams["axes.unicode_minus"] = False

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _fmt_p(p_val: float) -> str:
    """格式化 P 值."""
    if p_val < 0.001:
        return f"{p_val:.2e}"
    return f"{p_val:.3f}"


def _get_test_info(groups_data: list[np.ndarray]) -> str:
    """计算统计检验信息."""
    if len(groups_data) == 2:
        try:
            res_t = stats.ttest_ind(groups_data[0], groups_data[1], nan_policy="omit")
            res_u = stats.mannwhitneyu(groups_data[0], groups_data[1], alternative="two-sided")
            res_ks = stats.ks_2samp(groups_data[0], groups_data[1])
            # 使用 type ignore 解决 scipy 返回类型在某些检查器下的推断问题
            p_t = float(res_t.pvalue)  # type: ignore
            p_u = float(res_u.pvalue)  # type: ignore
            p_ks = float(res_ks.pvalue)  # type: ignore
            return f"\nT:{_fmt_p(p_t)} U:{_fmt_p(p_u)} KS:{_fmt_p(p_ks)}"
        except Exception:
            return "\nT:Err U:Err KS:Err"

    if len(groups_data) > 2:
        try:
            res_f = stats.f_oneway(*groups_data)
            res_h = stats.kruskal(*groups_data)
            p_f = float(res_f.pvalue)  # type: ignore
            p_h = float(res_h.pvalue)  # type: ignore
            return f"\nF:{_fmt_p(p_f)} K:{_fmt_p(p_h)}"
        except Exception:
            return "\nF:Err K:Err"

    return ""


def _draw_boxplot_on_ax(
    ax: Axes,
    df: pd.DataFrame,
    data_col: str,
    x_col: str | None,
    title: str,
    show_points: bool = True,
    show_means: bool = True,
    show_test: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs: Any,
) -> None:
    """在指定的 Axes 上绘制单个箱线图及其附属元素."""
    if df.empty:
        ax.set_visible(False)
        return

    # 1. 基础配置
    kwargs.setdefault("palette", "Set2")

    # 2. 将 x_col 转换为带零宽空格前缀的 Categorical 类型
    # 零宽空格(\u200b)视觉上不可见，但可以阻止 matplotlib 将字符串解析为数字
    # 从而避免 matplotlib.category 的 INFO 日志
    plot_df = df.copy()
    if x_col and x_col in plot_df.columns:
        str_values = plot_df[x_col].astype(str)
        unique_vals = str_values.unique()
        zwsp = "\u200b"
        new_cats = [f"{zwsp}{v}" for v in sorted(unique_vals)]
        mapping = {v: f"{zwsp}{v}" for v in unique_vals}
        plot_df[x_col] = pd.Categorical(str_values.map(mapping), categories=new_cats)

    # 3. 绘制 Boxplot (主体) - 设置 hue=x_col 并 legend=False 避免 FutureWarning
    sns.boxplot(data=plot_df, x=x_col, y=data_col, hue=x_col, ax=ax, showmeans=False, legend=False, **kwargs)

    # 调整透明度
    for patch in ax.patches:
        color = list(patch.get_facecolor())
        if len(color) >= 3:
            color = [*color[:3], 0.5]
            patch.set_facecolor(tuple(color))

    # 4. 绘制散点 (Stripplot)
    if show_points:
        sns.stripplot(data=plot_df, x=x_col, y=data_col, ax=ax, color=".25", size=4, alpha=0.6, jitter=True, zorder=3)

    # 4. 绘制均值点
    if show_means:
        _draw_means(ax, df, data_col, x_col)

    # 5. 统计检验
    test_info = ""
    if x_col and show_test:
        groups_data = [g[data_col].to_numpy() for _, g in df.groupby(x_col) if len(g) > 1]
        test_info = _get_test_info(groups_data)

    # 6. 设置标签和样式
    ax.set_title(f"{title}{test_info}")
    ax.set_xlabel(xlabel if xlabel is not None else (x_col if x_col else ""))
    ax.set_ylabel(ylabel if ylabel is not None else data_col)
    ax.grid(True, linestyle="--", alpha=0.5)
    sns.despine(ax=ax)


def _draw_means(ax: Axes, df: pd.DataFrame, data_col: str, x_col: str | None) -> None:
    """辅助绘制均值点."""
    if not x_col:
        mean_val = df[data_col].mean()
        ax.scatter(0, mean_val, marker="D", c="white", edgecolors="black", s=40, zorder=10)
        return

    means = df.groupby(x_col)[data_col].mean()
    if isinstance(df[x_col].dtype, pd.CategoricalDtype):
        means = means.reindex(df[x_col].cat.categories)
    else:
        means = means.sort_index()

    x_vals = np.arange(len(means))
    y_vals = means.to_numpy()
    mask = ~pd.isna(y_vals)

    if mask.any():
        ax.scatter(x_vals[mask], y_vals[mask], marker="D", c="white", edgecolors="black", s=40, zorder=10)


def plot_grouped_boxplot(
    data: pd.Series,
    groupby: pd.Series | pd.DataFrame | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] | None = None,
    show_points: bool = True,
    show_means: bool = True,
    show_test: bool = True,
    **kwargs: Any,
) -> "Figure":
    """绘制分组箱线图，支持 1-2 列分组."""
    # 1. 准备数据
    data_col = str(data.name) if data.name else "value"
    df, group_cols = _prepare_data(data, groupby, data_col)

    # 2. 确定布局
    fig, axes = _setup_layout(df, group_cols, figsize)

    # 3. 绘图
    if len(group_cols) <= 1:
        g_col = group_cols[0] if group_cols else None
        _draw_boxplot_on_ax(
            axes[0, 0],
            df,
            data_col,
            g_col,
            title or f"Boxplot of {data_col}",
            show_points,
            show_means,
            show_test,
            xlabel,
            ylabel,
            **kwargs,
        )
    else:
        _draw_two_column_grouping(
            fig, axes, df, data_col, group_cols, title, show_points, show_means, show_test, xlabel, ylabel, **kwargs
        )

    fig.tight_layout()
    return fig


def _prepare_data(data: pd.Series, groupby: Any, data_col: str) -> tuple[pd.DataFrame, list[str]]:
    """统一数据格式为 DataFrame."""
    if groupby is None:
        return pd.DataFrame({data_col: data}).dropna(), []

    if isinstance(groupby, pd.Series):
        g_col = str(groupby.name) if groupby.name else "group"
        df = pd.concat([data, groupby], axis=1).dropna()
        df.columns = [data_col, g_col]
        return df, [g_col]

    # DataFrame 情况
    raw_cols = [str(c) for c in groupby.columns][:2]
    df = pd.concat([data, groupby[raw_cols]], axis=1).dropna()
    df.columns = [data_col, *raw_cols]
    return df, raw_cols


def _setup_layout(df: pd.DataFrame, group_cols: list[str], figsize: tuple[int, int] | None) -> tuple[Any, Any]:
    """根据分组数量设置子图布局."""
    if len(group_cols) <= 1:
        n_groups = int(df[group_cols[0]].nunique()) if group_cols else 1
        size = figsize or (max(8, n_groups * 1.5), 6)
        fig, ax = plt.subplots(1, 1, figsize=size)
        return fig, np.array([[ax]])

    col1, col2 = group_cols
    u1, u2 = int(df[col1].nunique()), int(df[col2].nunique())
    ncols = max(u1, u2)
    size = figsize or (ncols * 4, 10)
    fig, axes = plt.subplots(2, ncols, figsize=size, squeeze=False)
    return fig, axes


def _draw_two_column_grouping(
    fig: Any,
    axes: Any,
    df: pd.DataFrame,
    data_col: str,
    group_cols: list[str],
    title: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """处理双列分组的绘图逻辑."""
    col1, col2 = group_cols
    u1, u2 = sorted(df[col1].unique()), sorted(df[col2].unique())
    ncols = axes.shape[1]

    # 第一行: 固定 col1, x轴为 col2
    for i, val in enumerate(u1):
        sub_df = df[df[col1] == val]
        if isinstance(sub_df, pd.DataFrame):
            _draw_boxplot_on_ax(axes[0, i], sub_df, data_col, col2, f"{col1}={val}", *args, **kwargs)
    for i in range(len(u1), ncols):
        axes[0, i].set_visible(False)

    # 第二行: 固定 col2, x轴为 col1
    for i, val in enumerate(u2):
        sub_df = df[df[col2] == val]
        if isinstance(sub_df, pd.DataFrame):
            _draw_boxplot_on_ax(axes[1, i], sub_df, data_col, col1, f"{col2}={val}", *args, **kwargs)
    for i in range(len(u2), ncols):
        axes[1, i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16)
