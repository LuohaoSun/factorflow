# FactorFlow

一个用于特征重要性分析的 Python 库。

## Core Design Principle

> `特征重要性分析`作为`特征工程`的一部分, 具体而言, 作为`特征选择`的**副作用**.

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

```

3. 特征保护与流式 API:

```python

```

4. 嵌入你自己的特征重要性分析方法:

只需保证你的特征重要性分析方法符合 TransformerMixin 规范, 即 fit 用于学习哪些列
应当保留, transform 用于返回保留的列(pd.DataFrame).

```python

```
