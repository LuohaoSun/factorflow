# FactorFlow 最佳实践指南

## 特征选择器开发规范

所有特征选择器**必须**继承 `src.factorflow.base.BaseSelector`。
该基类已经封装了 `sklearn.feature_selection.SelectorMixin`，并提供了针对 Pandas DataFrame 的增强支持、自动元数据管理以及特征保护机制。

### 1. 核心原则

1.  **继承 `BaseSelector`**:
    - **不要**直接继承 `SelectorMixin` 或 `BaseEstimator`。
    - **禁止**重写 `fit`（基类已提供统一的包装逻辑，子类应实现 `_fit`）。
    - **禁止**重写 `transform`（基类已标记为 `@final` 并提供实现）。
    - 基类会自动处理 Pandas DataFrame 的输入输出转换（保留列名）。

2.  **完善 `__init__`**:
    - 显式定义算法特有的参数。
    - **必须**在 `__init__` 中接收 `**kwargs` 并调用 `super().__init__(**kwargs)`。
    - **不需要**手动设置 `set_output(transform="pandas")`，基类默认已设置。
    - **严禁**使用 `*args`，必须显式定义所有算法参数，以保持 Sklearn 兼容性（`get_params`/`set_params`）。

3.  **专注于 `_fit` 和 `selected_features_`**:
    - **必须实现** `_fit(self, X, y, **kwargs)` 方法：计算特征重要性或统计量。
    - **必须赋值** `self.selected_features_`: 在 `_fit` 结束前，将算法选中的特征名称列表（`list[str]`）赋值给此属性。基类及回调（如特征保护）会自动处理最终结果。
    - **无需**实现 `_get_support_mask`，基类会根据 `selected_features_` 自动生成。

4.  **属性与兼容性**:
    - **自动获得**:
        - `self.feature_names_in_` / `self.n_features_in_`
        - `self.removed_features_`
    - **流式 API**:
        - `protect_features(pattern)`: 保护特定特征不被删除。
        - `check_features(pattern)`: 检查特征存在情况。
        - `check_selection()`: 检查特征数量变化。
        - `set_label(label)`: 设置日志打印时的标签。

### 2. 标准模版

```python
from typing import Any
import numpy as np
import pandas as pd
from src.factorflow.base import BaseSelector

class MySelector(BaseSelector):
    variances_: pd.Series

    def __init__(self, threshold: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def _fit(self, X: pd.DataFrame, y: Any = None, **kwargs) -> "MySelector":
        self.variances_ = X.var()
        mask = self.variances_ > self.threshold
        self.selected_features_ = X.columns[mask].tolist()
        return self
```

### 3. 如何使用

```python
selector = MySelector(threshold=0.5) \
    .set_label("VarianceFilter") \
    .protect_features("ID_Column") \
    .check_features("Critical_Feature*") \
    .check_selection()

# 训练与转换
df_cleaned = selector.fit_transform(df)

# 查看结果
print(f"Selected: {len(selector.selected_features_)}")
print(f"Removed: {selector.removed_features_}")
```
