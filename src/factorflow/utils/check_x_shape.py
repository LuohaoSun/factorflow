from collections.abc import Callable, Iterable
from typing import Literal

from loguru import logger
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def check_x_shape(
    label: str | None = None,
    *,
    print_shape: bool = True,
    print_columns: bool = False,
    max_columns_preview: int = 10,
    log_on: Literal["fit", "transform", "both"] = "fit",
    print_fn: Callable[[str], None] | None = None,
) -> "CheckXShape":
    """工厂函数：创建一个 CheckXShape 实例.

    函数在IDE中颜色与类不同, 用于在pipeline复杂时, 视觉上有效区分检查器与选择器.
    """
    return CheckXShape(
        label,
        print_shape=print_shape,
        print_columns=print_columns,
        max_columns_preview=max_columns_preview,
        log_on=log_on,
        print_fn=print_fn,
    )


class CheckXShape(BaseEstimator, TransformerMixin):
    """一个用于调试的Transformer，不修改数据，只打印当前数据的信息.

    可以插入到sklearn的pipeline中，用于观察在各个阶段数据的形状、列数量等。
    """

    def __init__(
        self,
        label: str | None = None,
        *,
        print_shape: bool = True,
        print_columns: bool = False,
        max_columns_preview: int = 10,
        log_on: Literal["fit", "transform", "both"] = "fit",
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        """一个用于调试的Transformer，不修改数据，只打印当前数据的信息.

        Args:
            label: 打印时的前缀标签，便于区分不同的节点。
            print_shape: 是否打印X的shape。
            print_columns: 是否打印列名（默认只打印数量，可设置此项为True打印预览）。
            max_columns_preview: 打印列名时最多展示多少个列。
            log_on: 在fit、transform或两者都打印。
            print_fn: 自定义打印函数，默认使用loguru.logger.info。

        """
        self.label = label
        self.print_shape = print_shape
        self.print_columns = print_columns
        self.max_columns_preview = max_columns_preview
        self.log_on = log_on
        self.print_fn = print_fn or logger.info

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> "CheckXShape":  # noqa: D102
        if self.log_on in ("fit", "both"):
            self._log(X, stage="fit")
        # Mark as fitted for sklearn compatibility
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        if self.log_on in ("transform", "both"):
            self._log(X, stage="transform")
        return X

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _log(self, X: pd.DataFrame, stage: str) -> None:
        parts: list[str] = []

        label = f"[{self.label}] " if self.label else ""
        parts.append(f"{label}{self.__class__.__name__} ({stage})")

        if self.print_shape:
            parts.append(f"shape={X.shape}")
            parts.append(f"n_columns={X.shape[1]}")

        if self.print_columns:
            preview = self._preview_columns(X.columns)
            parts.append(f"columns={preview}")

        message = " | ".join(parts)
        self.print_fn(message)

    def _preview_columns(self, columns: Iterable[str]) -> str:
        cols = list(columns)
        if not cols:
            return "[]"
        if len(cols) <= self.max_columns_preview:
            return f"{cols}"
        preview = cols[: self.max_columns_preview]
        return f"{preview} ... (total={len(cols)})"
