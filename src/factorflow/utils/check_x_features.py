from collections.abc import Callable, Iterable
import fnmatch
from typing import Literal

from loguru import logger
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def check_x_features(
    label: str | None = None,
    *,
    features: Iterable[str] | str,
    action: Literal["print", "warn", "raise"] = "print",
    log_on: Literal["fit", "transform", "both"] = "fit",
    print_fn: Callable[[str], None] | None = None,
) -> "CheckXFeatures":
    """工厂函数：创建一个 CheckXFeatures 实例.

    函数在IDE中颜色与类不同, 用于在pipeline复杂时, 视觉上有效区分检查器与选择器.
    """
    return CheckXFeatures(label, features=features, action=action, log_on=log_on, print_fn=print_fn)


class CheckXFeatures(BaseEstimator, TransformerMixin):
    """检查指定的特征是否在X中存在的Transformer.

    用于Pipeline中验证某些关键特征是否在之前的步骤中被保留或生成。
    例如, 如果你知道某个特征是正确答案, 这个检查器非常有用.
    支持精确匹配（传入列表）或通配符匹配（传入字符串）。
    """

    def __init__(
        self,
        label: str | None = None,
        *,
        features: Iterable[str] | str,
        action: Literal["print", "warn", "raise"] = "print",
        log_on: Literal["fit", "transform", "both"] = "fit",
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        """初始化.

        Args:
            features: 需要检查的特征.
                - 如果是字符串: 视为通配符模式 (Glob)，检查是否存在匹配该模式的特征.
                - 如果是列表: 视为特征名列表，检查这些特定特征是否都存在.
            action: 检查结果的处理方式.
                - "print": 仅打印检查结果.
                - "warn": 如果特征缺失（列表模式）或无匹配（通配符模式），打印警告.
                - "raise": 如果特征缺失（列表模式）或无匹配（通配符模式），抛出ValueError.
            log_on: 控制在哪个阶段执行检查 ("fit", "transform", "both"). 默认为 "fit".
            label: 打印时的标签.
            print_fn: 打印函数，默认为 loguru.logger.info.

        """
        self.features = features
        self.action = action
        self.log_on = log_on
        self.label = label
        self.print_fn = print_fn or logger.info

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> "CheckXFeatures":
        """Fit method."""
        if self.log_on in ("fit", "both"):
            self._check(X, stage="fit")
        # Mark as fitted for sklearn compatibility
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method."""
        if self.log_on in ("transform", "both"):
            self._check(X, stage="transform")
        return X

    def _check(self, X: pd.DataFrame, stage: str) -> None:
        prefix = f"[{self.label}] " if self.label else ""
        header = f"{prefix}{self.__class__.__name__} ({stage})"

        # 1. 通配符模式 (String)
        if isinstance(self.features, str):
            self._check_glob(X, header, self.features)
        # 2. 列表模式 (Iterable)
        else:
            self._check_list(X, header, list(self.features))

    def _check_glob(self, X: pd.DataFrame, header: str, pattern: str) -> None:
        # 使用 fnmatch 进行通配符匹配 (glob)
        matches = [c for c in X.columns if fnmatch.fnmatch(str(c), pattern)]

        if matches:
            if self.action == "print":
                # 展示前10个
                preview = matches[:10]
                more = "..." if len(matches) > 10 else ""
                self.print_fn(f"{header}: Found {len(matches)} features matching glob '{pattern}': {preview}{more}")
            return

        # No matches found
        msg = f"{header}: No features found matching glob '{pattern}'."
        self._handle_failure(msg)

    def _check_list(self, X: pd.DataFrame, header: str, target_features: list[str]) -> None:
        missing_features = [f for f in target_features if f not in X.columns]
        existing_features = [f for f in target_features if f in X.columns]

        if not missing_features:
            if self.action == "print":
                self.print_fn(f"{header}: All {len(target_features)} checked features are present.")
            return

        msg = (
            f"{header}: Missing {len(missing_features)}/{len(target_features)} features. "
            f"Missing: {missing_features}. "
            f"Present: {existing_features}."
        )
        self._handle_failure(msg)

    def _handle_failure(self, msg: str) -> None:
        if self.action == "raise":
            raise ValueError(msg)
        elif self.action == "warn":
            self.print_fn(f"WARNING: {msg}")
        else:  # print
            self.print_fn(msg)
