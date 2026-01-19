import fnmatch
from typing import Any

import pandas as pd

from ..base import Selector


class SelectFeatureName(Selector):
    """根据特征名称选择特征."""

    def __init__(self, feature_name: str | list[str], *, _exclude: bool = False, **kwargs):
        """根据特征名称选择特征. 此选择器的反向操作 `FilterFeatureName` 可能更常用.

        Args:
            feature_name: 特征名称, 支持glob匹配, 例如"TOTAL_*"表示选择所有以"TOTAL_"开头的特征.
                也可以传入一个包含多个glob模式的列表, 例如 ["TOTAL_*", "AVG_*"].
            _exclude: (内部参数) 是否排除匹配的特征.
                默认为 False (保留匹配的特征).
                设置为 True 时行为反转 (移除匹配的特征), 仅供 FilterFeatureName 子类使用.
                普通用户不应直接使用此参数, 请使用 FilterFeatureName 类代替.
            **kwargs: 传递给父类的参数.
        """
        super().__init__(**kwargs)
        self.feature_name = feature_name
        self._exclude = _exclude

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "SelectFeatureName":
        """拟合选择器."""
        # 计算匹配列表
        patterns = [self.feature_name] if isinstance(self.feature_name, str) else self.feature_name

        matches = [name for name in self.feature_names_in_ if any(fnmatch.fnmatch(str(name), p) for p in patterns)]

        if self._exclude:
            # 如果是 exclude 模式，保留不匹配的
            matches_set = set(matches)
            self.selected_features_ = [name for name in self.feature_names_in_ if name not in matches_set]
        else:
            # 如果是 select 模式，保留匹配的
            self.selected_features_ = matches

        return self


class FilterFeatureName(SelectFeatureName):
    """根据特征名称过滤特征."""

    def __init__(self, feature_name: str | list[str], **kwargs):
        """根据特征名称过滤特征. SelectFeatureName的反向操作.

        Args:
            feature_name: 特征名称, 支持glob匹配, 例如"TOTAL_*"表示过滤掉所有以"TOTAL_"开头的特征.
                也可以传入一个包含多个glob模式的列表, 例如 ["TOTAL_*", "AVG_*"].
            **kwargs: 传递给父类的参数.
        """
        super().__init__(feature_name, _exclude=True, **kwargs)
