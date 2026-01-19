import pandas as pd
import pytest
from factorflow.simple_selectors.select_feature_name import SelectFeatureName, FilterFeatureName

def test_select_feature_name(sample_data):
    X, y = sample_data
    
    # Test single name
    selector = SelectFeatureName(feature_name="feature_1")
    selector.fit(X, y)
    assert selector.selected_features_ == ["feature_1"]
    X_tr = selector.transform(X)
    assert X_tr.columns.tolist() == ["feature_1"]
    
    # Test glob pattern
    selector = SelectFeatureName(feature_name="feature_*")
    selector.fit(X, y)
    assert set(selector.selected_features_) == {"feature_1", "feature_2", "feature_3", "feature_4"}
    
    # Test list of patterns
    selector = SelectFeatureName(feature_name=["feature_1", "name_prefix_*"])
    selector.fit(X, y)
    assert set(selector.selected_features_) == {"feature_1", "name_prefix_1", "name_prefix_2"}

def test_filter_feature_name(sample_data):
    X, y = sample_data
    
    # Test single name
    selector = FilterFeatureName(feature_name="feature_1")
    selector.fit(X, y)
    assert "feature_1" not in selector.selected_features_
    assert len(selector.selected_features_) == len(X.columns) - 1
    
    # Test glob pattern
    selector = FilterFeatureName(feature_name="feature_*")
    selector.fit(X, y)
    assert not any(f.startswith("feature_") for f in selector.selected_features_)
    
    # Test list of patterns
    selector = FilterFeatureName(feature_name=["feature_*", "target_col"])
    selector.fit(X, y)
    remaining = set(selector.selected_features_)
    assert not any(f.startswith("feature_") for f in remaining)
    assert "target_col" not in remaining
    assert "name_prefix_1" in remaining

def test_selector_base_features(sample_data):
    """Test features inherited from BaseSelector like protection."""
    X, y = sample_data
    
    # Protect a feature that would otherwise be filtered out
    selector = FilterFeatureName(feature_name="feature_1")
    selector.protect_features("feature_1")
    selector.fit(X, y)
    assert "feature_1" in selector.selected_features_
    
    # Check features callback (smoke test)
    selector = SelectFeatureName(feature_name="feature_1")
    selector.check_features("feature_1")
    selector.check_selection()
    selector.fit(X, y) # Should not raise error
