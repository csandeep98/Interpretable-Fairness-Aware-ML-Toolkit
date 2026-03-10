"""
Test suite for counterfactuals module.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import counterfactuals


class TestCounterfactualGenerator:
    """Test CounterfactualGenerator class."""
    
    def setup_method(self):
        """Setup test data and model."""
        # Create synthetic dataset
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3, 
            n_redundant=2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.X_background = X
        self.y_background = y
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Test instance
        self.test_instance = X[0]
        self.test_prediction = y[0]
    
    def test_generator_initialization(self):
        """Test counterfactual generator initialization."""
        generator = counterfactuals.CounterfactualGenerator(
            model=self.model,
            X_background=self.X_background,
            y_background=self.y_background,
            feature_names=self.feature_names
        )
        
        assert generator.model == self.model
        assert generator.X_background.shape == self.X_background.shape
        assert generator.feature_names == self.feature_names
    
    def test_generate_counterfactual(self):
        """Test counterfactual generation."""
        generator = counterfactuals.CounterfactualGenerator(
            model=self.model,
            X_background=self.X_background,
            y_background=self.y_background,
            feature_names=self.feature_names
        )
        
        cf_result = generator.generate_counterfactual(
            X=self.test_instance,
            y_original=self.test_prediction,
            max_distance=2.0,
            num_candidates=3
        )
        
        assert isinstance(cf_result, dict)
        assert 'original_instance' in cf_result
        assert 'original_prediction' in cf_result
        assert 'target_class' in cf_result
        assert 'counterfactuals' in cf_result
        assert 'num_found' in cf_result
        
        # Check that we found at least one counterfactual
        assert cf_result['num_found'] >= 1
        assert len(cf_result['counterfactuals']) >= 1
    
    def test_protected_features(self):
        """Test counterfactual generation with protected features."""
        generator = counterfactuals.CounterfactualGenerator(
            model=self.model,
            X_background=self.X_background,
            y_background=self.y_background,
            feature_names=self.feature_names,
            protected_features=['feature_0']  # Don't modify feature_0
        )
        
        cf_result = generator.generate_counterfactual(
            X=self.test_instance,
            y_original=self.test_prediction,
            max_distance=2.0,
            num_candidates=1
        )
        
        if cf_result['counterfactuals']:
            cf = cf_result['counterfactuals'][0]
            changes = cf['changes']
            
            # feature_0 should not have changed
            assert abs(changes['feature_0']['difference']) < 1e-10
    
    def test_counterfactual_quality_evaluation(self):
        """Test counterfactual quality evaluation."""
        generator = counterfactuals.CounterfactualGenerator(
            model=self.model,
            X_background=self.X_background,
            y_background=self.y_background,
            feature_names=self.feature_names
        )
        
        cf_result = generator.generate_counterfactual(
            X=self.test_instance,
            y_original=self.test_prediction,
            max_distance=2.0,
            num_candidates=3
        )
        
        if cf_result['counterfactuals']:
            quality_metrics = generator.evaluate_counterfactual_quality(
                cf_result['counterfactuals']
            )
            
            assert isinstance(quality_metrics, dict)
            assert 'validity' in quality_metrics
            assert 'proximity' in quality_metrics
            assert 'sparsity' in quality_metrics
            
            # Check that metrics are in valid ranges
            assert 0 <= quality_metrics['validity'] <= 1
            assert 0 <= quality_metrics['proximity'] <= 1
            assert 0 <= quality_metrics['sparsity'] <= 1
    
    def test_batch_generate_counterfactuals(self):
        """Test batch counterfactual generation."""
        generator = counterfactuals.CounterfactualGenerator(
            model=self.model,
            X_background=self.X_background,
            y_background=self.y_background,
            feature_names=self.feature_names
        )
        
        X_batch = self.X_background[:3]
        y_batch = self.y_background[:3]
        
        batch_results = generator.batch_generate_counterfactuals(
            X_batch, y_batch, max_distance=2.0, num_candidates=1
        )
        
        assert isinstance(batch_results, list)
        assert len(batch_results) == 3
        
        for result in batch_results:
            assert isinstance(result, dict)
            assert 'counterfactuals' in result


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Setup test data."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=3, 
            n_redundant=2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.X = X
        self.y = y
    
    def test_generate_counterfactuals_function(self):
        """Test convenience generate_counterfactuals function."""
        result = counterfactuals.generate_counterfactuals(
            model=self.model,
            X=self.X[0],
            y=self.y[0],
            X_background=self.X,
            y_background=self.y,
            feature_names=[f'feature_{i}' for i in range(self.X.shape[1])]
        )
        
        assert isinstance(result, dict)
        assert 'counterfactuals' in result


if __name__ == "__main__":
    pytest.main([__file__])
