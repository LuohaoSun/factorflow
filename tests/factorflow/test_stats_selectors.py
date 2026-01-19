import numpy as np
import pandas as pd
import pytest
from factorflow.stats_selectors.select_na_ratio import SelectNARatio
from factorflow.stats_selectors.select_constant_ratio import SelectConstantRatio
from factorflow.stats_selectors.select_collinearity import SelectCollinearity
from factorflow.stats_selectors.select_vif import SelectVif
from factorflow.stats_selectors.select_fpr_na_safe import (
    SelectFprAnova, SelectFprPearson, SelectFprChi2, SelectFprKS, SelectFprAnovaReverse
)
from factorflow.stats_selectors.select_kbest_na_safe import (
    SelectKBestMutualInfo, SelectKBestAnova, SelectKBestChi2
)
from factorflow.stats_selectors.select_percentile_na_safe import (
    SelectPercentileMutualInfo, SelectPercentileAnova, SelectPercentileChi2
)

def test_select_na_ratio(sample_data):
    X, y = sample_data
    # na_col has 2/5 = 0.4 NA, all_na_col has 1.0 NA
    
    # Test ratio
    selector = SelectNARatio(na_threshold=0.5, strategy="ratio")
    selector.fit(X, y)
    assert "na_col" in selector.selected_features_
    assert "all_na_col" not in selector.selected_features_
    
    selector = SelectNARatio(na_threshold=0.3, strategy="ratio")
    selector.fit(X, y)
    assert "na_col" not in selector.selected_features_
    
    # Test count
    selector = SelectNARatio(na_threshold=1, strategy="count")
    selector.fit(X, y)
    assert "na_col" not in selector.selected_features_ # has 2 NAs

def test_select_constant_ratio(sample_data):
    X, y = sample_data
    # feature_3 is constant [1, 1, 1, 1, 1]
    
    selector = SelectConstantRatio(strategy="constant")
    selector.fit(X, y)
    assert "feature_3" not in selector.selected_features_
    assert "feature_1" in selector.selected_features_
    
    # Test quasi-constant
    X_quasi = X.copy()
    X_quasi["quasi"] = [1, 1, 1, 1, 0] # 0.8 ratio
    selector = SelectConstantRatio(threshold=0.7, strategy="quasi_constant")
    selector.fit(X_quasi, y)
    assert "quasi" not in selector.selected_features_

def test_select_collinearity(collinear_data):
    X, y = collinear_data
    # feat_A and feat_B are perfectly correlated
    
    selector = SelectCollinearity(threshold=0.9)
    selector.fit(X, y)
    # One of feat_A or feat_B should be dropped
    assert not ("feat_A" in selector.selected_features_ and "feat_B" in selector.selected_features_)
    assert len(selector.selected_features_) < len(X.columns)
    
    groups = selector.get_feature_groups()
    assert len(groups) > 0

def test_select_vif(collinear_data):
    X, y = collinear_data
    
    selector = SelectVif(threshold=5.0)
    selector.fit(X, y)
    # Perfectly collinear features should be removed
    assert not ("feat_A" in selector.selected_features_ and "feat_B" in selector.selected_features_)
    assert len(selector.selected_features_) < len(X.columns)

def test_select_fpr(sample_data):
    X, y = sample_data
    # Use a subset of features that are numeric and have no NA for simplicity in some tests
    X_num = X[["feature_1", "feature_2", "feature_4"]]
    
    # ANOVA
    selector = SelectFprAnova(alpha=0.05)
    selector.fit(X_num, y)
    assert hasattr(selector, "pvalues_")
    
    # Pearson
    selector = SelectFprPearson(alpha=0.05)
    selector.fit(X_num, y)
    assert hasattr(selector, "pvalues_")
    
    # KS
    selector = SelectFprKS(alpha=0.05)
    selector.fit(X_num, y)
    assert hasattr(selector, "pvalues_")
    
    # Chi2 (requires non-negative)
    X_pos = X_num + 10
    selector = SelectFprChi2(alpha=0.05)
    selector.fit(X_pos, y)
    assert hasattr(selector, "pvalues_")

    # ANOVA Reverse (categorical feature, continuous target)
    X_cat = pd.DataFrame({
        "cat_1": [0, 0, 1, 1, 2],
        "cat_2": [0, 1, 0, 1, 0]
    })
    y_cont = pd.Series([1.0, 1.1, 5.0, 5.2, 10.0])
    selector = SelectFprAnovaReverse(alpha=0.05)
    selector.fit(X_cat, y_cont)
    assert hasattr(selector, "pvalues_")

def test_select_kbest(sample_data):
    X, y = sample_data
    X_num = X[["feature_1", "feature_2", "feature_4"]]
    
    # ANOVA
    selector = SelectKBestAnova(k=2)
    selector.fit(X_num, y)
    assert len(selector.selected_features_) == 2
    
    # Mutual Info
    selector = SelectKBestMutualInfo(k=1)
    selector.fit(X_num, y)
    assert len(selector.selected_features_) == 1
    
    # Chi2
    X_pos = X_num + 10
    selector = SelectKBestChi2(k=1)
    selector.fit(X_pos, y)
    assert len(selector.selected_features_) == 1

def test_select_percentile(sample_data):
    X, y = sample_data
    # Create data where features have clearly different scores
    # Ensure at least 2 samples per class for ANOVA
    X_distinct = pd.DataFrame({
        "feat_high": [1, 1, 5, 5, 10, 10],
        "feat_mid":  [1, 2, 1, 2, 1, 2],
        "feat_low":  [1, 1, 1, 1, 1, 1]
    })
    y_distinct = pd.Series([0, 0, 1, 1, 2, 2])
    
    # ANOVA - select top 34% (should be 1 feature out of 3)
    selector = SelectPercentileAnova(percentile=34)
    selector.fit(X_distinct, y_distinct)
    assert len(selector.selected_features_) == 1
    assert selector.selected_features_ == ["feat_high"]
