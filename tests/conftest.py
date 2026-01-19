import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [5, 4, 3, 2, 1],
            "feature_3": [1, 1, 1, 1, 1],  # Constant
            "feature_4": [1, 2, 1, 2, 1],
            "target_col": [0, 1, 0, 1, 0],
            "name_prefix_1": [10, 20, 30, 40, 50],
            "name_prefix_2": [1, 2, 3, 4, 5],
            "na_col": [1, np.nan, 3, np.nan, 5],
            "all_na_col": [np.nan] * 5,
        }
    )
    y = pd.Series([0, 1, 0, 1, 0], name="target")
    return X, y


@pytest.fixture
def collinear_data():
    """Create a sample DataFrame with collinear features."""
    X = pd.DataFrame(
        {
            "feat_A": [1, 2, 3, 4, 5],
            "feat_B": [1, 2, 3, 4, 5],  # Perfectly correlated with A
            "feat_C": [5, 4, 3, 2, 1],  # Perfectly negatively correlated with A
            "feat_D": [1, 0, 1, 0, 1],
            "feat_E": [1, 2, 3, 4, 100],  # Outlier
        }
    )
    y = pd.Series([0, 1, 0, 1, 0], name="target")
    return X, y
