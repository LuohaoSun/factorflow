import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from factorflow.xai_selectors.select_from_model_shap_cv import SelectFromModelShapCV
from factorflow.xai_selectors.select_from_model_shap_null_importance import SelectFromModelShapNullImportance

def test_select_from_model_shap_cv(sample_data):
    X, y = sample_data
    # Filter to numeric features for the model
    X_num = X[["feature_1", "feature_2", "feature_4"]]
    
    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    selector = SelectFromModelShapCV(
        estimator=estimator,
        task_type="classification",
        n_features_to_select=2,
        n_splits=2,
        verbose=0
    )
    
    selector.fit(X_num, y)
    assert len(selector.selected_features_) == 2
    assert hasattr(selector, "shap_values_oof_")
    assert selector.shap_values_oof_.shape == X_num.shape

def test_select_from_model_shap_null_importance(sample_data):
    X, y = sample_data
    X_num = X[["feature_1", "feature_2", "feature_4"]]
    
    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    # Using small n_trials for speed in tests
    selector = SelectFromModelShapNullImportance(
        estimator=estimator,
        task_type="classification",
        n_trials=5,
        cv=2,
        verbose=0,
        mode="p_value",
        threshold=1.0 # Keep all for testing
    )
    
    selector.fit(X_num, y)
    assert len(selector.selected_features_) > 0
    assert hasattr(selector, "p_values_")
    assert len(selector.p_values_) == X_num.shape[1]
