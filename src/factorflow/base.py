from abc import abstractmethod
import contextlib
from datetime import datetime
import fnmatch
import inspect
import re
from typing import Any, Literal, final
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class Callback:
    """Base class for FactorFlow Selectors callbacks."""

    def on_fit_start(self, selector: "Selector", X: pd.DataFrame, y: Any = None) -> None:
        """Handle the beginning of fit."""
        pass

    def on_fit_end(self, selector: "Selector", X: pd.DataFrame, y: Any = None) -> None:
        """Handle the end of fit."""
        pass


class CheckXShape(Callback):
    """Callback to log shape of X before and after fit."""

    def on_fit_start(self, selector: "Selector", X: pd.DataFrame, y: Any = None) -> None:
        """Log shape before fit."""
        logger.info(f"[{selector.label}] (pre_fit) X Shape Check: {X.shape}")

    def on_fit_end(self, selector: "Selector", X: pd.DataFrame, y: Any = None) -> None:
        """Log shape after fit."""
        # Note: We calculate output shape based on selected features
        try:
            n_features_out = len(selector.get_feature_names_out())
            out_shape = (X.shape[0], n_features_out)
            logger.info(f"[{selector.label}] (post_fit) X Shape Check: {out_shape}")
        except Exception as e:
            logger.warning(f"[{selector.label}] (post_fit) Could not determine output shape: {e}")


class CheckFeatures(Callback):
    """Callback to check for presence of specific features."""

    def __init__(self, patterns: list[str] | None = None):
        """Initialize FeatureCheckCallback.

        Args:
        ----
            patterns: List of glob patterns to check.
        """
        self.patterns = patterns or []

    def add_patterns(self, patterns: str | list[str]) -> None:
        """Add more patterns to check.

        Args:
        ----
            patterns: A single pattern or list of patterns.
        """
        if isinstance(patterns, str):
            self.patterns.append(patterns)
        else:
            self.patterns.extend(patterns)

    def on_fit_start(self, selector: "Selector", X: pd.DataFrame, y: Any = None) -> None:
        """Check features before fit."""
        self._check(selector, "pre_fit", X.columns.tolist())

    def on_fit_end(self, selector: "Selector", X: pd.DataFrame, y: Any = None) -> None:
        """Check features after fit."""
        # We assume selected_features_ is available after fit
        with contextlib.suppress(Exception):
            self._check(selector, "post_fit", selector.selected_features_)

    def _check(self, selector: "Selector", stage: Literal["pre_fit", "post_fit"], current_features: list[str]) -> None:
        if not self.patterns:
            return

        feat_list = [str(f) for f in current_features]
        summary = []

        for pattern in self.patterns:
            matches = fnmatch.filter(feat_list, pattern)
            count = len(matches)
            if count == 0:
                summary.append(f"❌ '{pattern}': 0 matches")
            else:
                preview = matches[:3]
                preview_str = ", ".join(preview) + ("..." if count > 3 else "")
                summary.append(f"✅ '{pattern}': {count} matches ({preview_str})")

        if summary:
            logger.info(f"[{selector.label}] ({stage}) Feature Checks: " + "; ".join(summary))


class ProtectFeatures(Callback):
    """Callback to identify protected features."""

    def __init__(self, patterns: list[str] | None = None):
        """Initialize FeatureProtectionCallback.

        Args:
        ----
            patterns: List of glob patterns to protect.
        """
        self.patterns = patterns or []

    def add_patterns(self, patterns: str | list[str]) -> None:
        """Add more patterns to protect.

        Args:
        ----
            patterns: A single pattern or list of patterns.
        """
        if isinstance(patterns, str):
            self.patterns.append(patterns)
        else:
            self.patterns.extend(patterns)

    def on_fit_start(self, selector: "Selector", X: pd.DataFrame, y: Any = None) -> None:
        """Calculate protected features based on input columns and patterns."""
        pass

    def on_fit_end(self, selector: "Selector", X: pd.DataFrame, y: Any = None) -> None:
        """Apply protection by adding protected features to selector.selected_features_."""
        if not self.patterns:
            return

        feature_names_in = X.columns.tolist()
        # Create regex from glob patterns
        regex_pattern = "|".join([fnmatch.translate(p) for p in self.patterns])
        matcher = re.compile(regex_pattern)

        protected_features = [col for col in feature_names_in if matcher.match(str(col))]

        if protected_features:
            current_selected = set(selector.selected_features_)
            new_selected = list(current_selected.union(protected_features))
            # Keep original order if possible, or just sort to be deterministic
            # Sorting might be safer for downstream consistency
            new_selected.sort(key=lambda x: feature_names_in.index(x))

            added_count = len(set(protected_features) - current_selected)
            if added_count > 0:
                logger.info(
                    f"[{selector.label}] Protected {added_count} features added to selection: "
                    f"{list(set(protected_features) - current_selected)}"
                )

            selector.selected_features_ = new_selected


class Selector(BaseEstimator, SelectorMixin):
    """特征选择器基类.

    旨在简化自定义特征选择器的开发，提供统一的接口和增强功能。
    它继承自 `sklearn.feature_selection.SelectorMixin`，并默认输出 Pandas DataFrame。

    核心功能:
    1. **特征保护**: 通过 `protect_features` 方法，可以强制保留某些特征（支持 glob 模式），即使它们被算法判定为应剔除。
    2. **特征检查**: 通过 `check_features` 方法，可以在 fit 前后自动检查特征是否存在。
    3. **形状校验**: 通过 `check_selection` 方法，可以在 fit 前后自动打印特征数量的变化。
    4. **链式调用**: 通过流式调用，可以方便地配置选择器，
       如 `selector.protect_features("id").check_features("feature*").check_selection()`。
    5. **sklearn 兼容性**: 完全兼容 `get_params()`, `set_params()`。
    6. **Callback 机制**: 支持自定义回调函数，在 fit 的不同阶段执行逻辑。

    子类开发契约:
    - **必须实现** `_fit(self, X, y, **kwargs)` 方法。不要重写 `fit`。
    - **必须实现** `selected_features_` 属性 (property)，返回保留的特征名称列表。
    - **必须**在 `__init__` 中接收 `**kwargs` 并调用 `super().__init__(**kwargs)` 以保持兼容性。
    - **无需**实现 `_get_support_mask`，基类已提供基于 `selected_features_` 和保护逻辑的实现。
    - **禁止**重写 `fit` 和 `transform`。

    Attributes
    ----------
        selected_features_ (list[str]): 拟合后确定的最终特征列表。
        feature_names_in_ (np.ndarray): 输入特征名称。
        n_features_in_ (int): 输入特征数量。
    """

    # 类型提示
    selected_features_: list[str]
    feature_names_in_: np.ndarray
    n_features_in_: int
    _input_shape: tuple[int, ...]
    callbacks: list[Callback]

    def __init__(
        self,
        *,
        protected_features_patterns: list[str] | None = None,
        selection_check: bool = False,
        check_features_patterns: list[str] | None = None,
        label: str | None = None,
        callbacks: list[Callback] | None = None,
        **kwargs,
    ):
        """初始化 BaseSelector.

        Args:
        ----
            protected_features_patterns: 即使被算法剔除也要强制保留的特征模式列表.
            selection_check: 是否在 fit 前后检查特征数量变化.
            check_features_patterns: 需要检查的特征模式列表.
            label: 此选择器的标签, 用于在打印检查结果时显示.
            callbacks: 自定义回调列表.
            **kwargs: 接收多余参数以保持兼容性.
        """
        super().__init__()
        self.set_output(transform="pandas")
        self.protected_features_patterns = protected_features_patterns
        self.selection_check = selection_check
        self.check_features_patterns = check_features_patterns
        self.label = label
        self.callbacks = callbacks or []

        # 初始化基于参数的 Callbacks
        self._init_param_callbacks()

    def _init_param_callbacks(self):
        """将构造函数参数转换为 Callbacks."""
        if self.selection_check:
            self._ensure_callback(CheckXShape)

        if self.check_features_patterns:
            cb = self._ensure_callback(CheckFeatures)
            cb.add_patterns(self.check_features_patterns)

        if self.protected_features_patterns:
            cb = self._ensure_callback(ProtectFeatures)
            cb.add_patterns(self.protected_features_patterns)

    def _ensure_callback(self, callback_cls: type) -> Any:
        """确保特定类型的 Callback 存在，如果不存在则创建并添加."""
        for cb in self.callbacks:
            if isinstance(cb, callback_cls):
                return cb
        new_cb = callback_cls()
        self.callbacks.append(new_cb)
        return new_cb

    def _remove_callback(self, callback_cls: type):
        """移除特定类型的 Callback."""
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_cls)]

    # ============================== abstract methods ==============================
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "Selector":
        """子类实现的具体拟合逻辑.

        在此方法中，子类应计算并将算法选中的特征赋值给 `self.selected_features_`。

        Args:
        ----
            X: 输入特征 DataFrame.
            y: 目标变量.
            **kwargs: 额外参数.

        Returns:
        -------
            self
        """
        ...

    # ============================== sklearn compatibility ==============================
    @final
    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: D102
        params = super().get_params(deep=deep)
        base_init_signature = inspect.signature(Selector.__init__)
        base_params = {
            name: getattr(self, name) for name in base_init_signature.parameters if name not in ("self", "kwargs")
        }
        params.update(base_params)
        return params

    @final
    def get_feature_names_in(self) -> np.ndarray:
        """获取输入特征名称列表."""
        return np.array(self.feature_names_in_)

    # ============================== fluent API ==============================
    def set_label(self, label: str) -> "Selector":
        """设置此选择器的标签, 用于在打印检查结果时显示.

        Args:
        ----
            label: 此选择器的标签.
        """
        self.label = label
        return self

    def check_selection(self, selection_check: bool = True) -> "Selector":
        """设置此选择器需要在 fit 前后检查特征数量变化."""
        self.selection_check = selection_check
        if selection_check:
            self._ensure_callback(CheckXShape)
        else:
            self._remove_callback(CheckXShape)
        return self

    def check_features(
        self,
        feature_pattern: str | list[str],
    ) -> "Selector":
        """设置此选择器需要在 fit 前后检查哪些特征.

        Args:
        ----
            feature_pattern: 需要检查的特征名称. 支持 glob 模式.
        """
        if isinstance(feature_pattern, list):
            for pattern in feature_pattern:
                self.check_features(pattern)
            return self

        if hasattr(self, "feature_names_in_"):
            warnings.warn("Calling check_features after fit() has no immediate effect.", UserWarning, stacklevel=2)

        # Update param for persistence
        self.check_features_patterns = self.check_features_patterns or []
        self.check_features_patterns.append(feature_pattern)

        # Update callback
        cb = self._ensure_callback(CheckFeatures)
        cb.add_patterns(feature_pattern)
        return self

    def protect_features(self, feature_pattern: str | list[str]) -> "Selector":
        """设置此选择器需要保护哪些特征不被过滤掉.

        Args:
        ----
            feature_pattern: 需要保护的特征名称. 支持 glob 模式.
        """
        if isinstance(feature_pattern, list):
            for pattern in feature_pattern:
                self.protect_features(pattern)
            return self

        if hasattr(self, "feature_names_in_"):
            warnings.warn("Calling protect_features after fit() has no immediate effect.", UserWarning, stacklevel=2)

        # Update param for persistence
        self.protected_features_patterns = self.protected_features_patterns or []
        current = set(self.protected_features_patterns)
        current.add(feature_pattern)
        self.protected_features_patterns = list(current)

        # Update callback
        cb = self._ensure_callback(ProtectFeatures)
        cb.add_patterns(feature_pattern)
        return self

    # ============================== core functions & properties ==============================
    @final
    def fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "Selector":  # noqa: D102
        self.label = self.label or f"{self.__class__.__name__}_{datetime.now().strftime('%H%M%S_%f')}"

        if isinstance(y, pd.DataFrame):
            logger.warning(
                f"[{self.label}] Target variable y is a DataFrame. "
                f"Automatically converting to Series using the first column: '{y.columns[0]}'."
            )
            y = y.iloc[:, 0]

        self.feature_names_in_ = np.array(X.columns.tolist())
        self.n_features_in_ = len(self.feature_names_in_)
        self._input_shape = X.shape

        # --- Pre-fit Callbacks ---
        for cb in self.callbacks:
            cb.on_fit_start(self, X, y)

        self._fit(X, y, **kwargs)

        # Ensure selected_features_ is set by _fit
        if not hasattr(self, "selected_features_"):
            raise AttributeError(f"[{self.label}] _fit() must set self.selected_features_")

        # --- Post-fit Callbacks ---
        for cb in self.callbacks:
            cb.on_fit_end(self, X, y)

        return self

    @final
    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:  # noqa: D102
        return super().get_feature_names_out(input_features=input_features)

    @final
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        return super().transform(X)  # pyright: ignore[reportReturnType]

    @final
    def _get_support_mask(self) -> np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        selected_set = set(self.selected_features_)
        mask = np.array([(str(f) in selected_set) for f in self.feature_names_in_])
        return mask

    @property
    def removed_features_(self) -> list[str]:
        """获取被移除的特征名称列表."""
        check_is_fitted(self, "feature_names_in_")
        return self.get_feature_names_in()[~self._get_support_mask()].tolist()


# Alias for backward compatibility
BaseSelector = Selector
