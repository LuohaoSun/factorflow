from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder


def make_onehot_encoder(dtype_include: list[type] | None = None) -> ColumnTransformer:
    """创建用于对指定类型的特征列进行独热编码（OneHot Encoding）的ColumnTransformer。

    Args:
        dtype_include (list[type], optional): 需要进行独热编码的列的数据类型列表。
            默认为 [object]，即只转换字符串/类别特征。

    Returns:
        ColumnTransformer: 对指定类型列进行独热编码，其余特征保持不变（passthrough），
        输出为pandas DataFrame格式。
    """
    dtype_include = dtype_include or [object]

    return make_column_transformer(
        (OneHotEncoder(sparse_output=False), make_column_selector(dtype_include=dtype_include)),
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
