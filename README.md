# FactorFlow

一个用于特征重要性分析的 Python 库。

## Core Design Principle

> `特征重要性分析`作为`特征工程`的一部分, 具体而言, 作为`特征选择`的**副作用**, 并全面采用 sklearn 规范, 从而实现复用性和可扩展性等好处.

## Installation

```bash
pip install git+https://github.com/luohaosun/factorflow.git
```

## Examples

1. 使用五折交叉验证计算 shap 值, 并选择重要性最高的 2 个特征:

```python
from factorflow import SelectFromModelShapCV

X = ...
y = ...
estimator = XGBClassifier()
selector = SelectFromModelShapCV(
    estimator=estimator,
    task_type="classification",
    n_features_to_select=2,
    verbose=2, # 设置为2以获取可视化结果
)
selector.fit_transform(X, y)
```

2. 使用 Pipeline 搭建复杂流程:

```python
from factorflow import make_onehot_encoder, SelectFprAnova, SelectFromModelShapCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

X = ...
y = ...
estimator = XGBClassifier()
fe_pipeline = make_pipeline(
    make_onehot_encoder(),
    SelectFprAnova(0.05),
    SelectFromModelShapCV(
        estimator=estimator,
        task_type="classification",
        n_features_to_select=2,
    ),
)

fe_pipeline.fit_transform(X, y)
```

3. 特征保护与流式 API:

```python
fe_pipeline = make_pipeline(
    make_onehot_encoder(),
    SelectFprAnova(0.05).check_selection().protect_features("feat_1"),
    SelectFromModelShapCV(...).check_selection().check_features("feat_*"),
)
```

- `.check_selection()` 会在 fit 前后检查特征数量变化.
- `.protect_features("feat_1")` 会保护 feat_1 特征不被过滤掉, 即使其 p 值大于 0.05.
- `.check_features("feat_*")` 会在 fit 前后检查 `feat_*` (glob 模式) 特征是否存在.

4. 嵌入你自己的特征重要性分析方法:

只需保证你的特征重要性分析方法符合 TransformerMixin 规范, 即 fit 用于学习哪些列应当保留, transform 用于返回保留的列(pd.DataFrame).

```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

class MySelector(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.iloc[:, :2]

fe_pipeline = make_pipeline(
    MySelector(),
    SelectFromModelShapCV(...),
)

fe_pipeline.fit_transform(X, y)
```

5. 使用 Callback 扩展功能

```python
class PlotYDist:
    """Plot distribution of the target variable."""

    def on_fit_start(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        """Plot target distribution before fit."""
        if y is None:
            return

        fig, ax = plt.subplots()
        sns.histplot(y, kde=True, ax=ax)
        ax.set_title("Target Distribution")
        plt.show()

    def on_fit_end(self, selector: Selector, X: pd.DataFrame, y: Any = None) -> None:
        ...

selector = SelectFromModelShapCV(...).add_callback(PlotYDist())
selector.fit_transform(X, y)
```
