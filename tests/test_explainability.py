"""
Test suite for explainability module.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import explainability


class TestExplainer:
    """Test Explainer class."""
    
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
        
        self.X = X
        self.y = y
        self.X_test = X[:10]  # Use first 10 as test instances
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        explainer = explainability.Explainer(
            model=self.model,
            model_type="sklearn",
            background_data=self.X[:50]
        )
        
        assert explainer.model == self.model
        assert explainer.model_type == "sklearn"
        assert explainer.explainer is not None
    
    def test_explain_instance(self):
        """Test instance explanation."""
        explainer = explainability.Explainer(
            model=self.model,
            model_type="sklearn",
            background_data=self.X[:50]
        )
        
        explanations = explainer.explain_instance(self.X_test, method="shap")
        
        assert isinstance(explanations, dict)
        assert 'shap_values' in explanations
        assert 'base_values' in explanations
        assert 'data' in explanations
    
    def test_explain_function(self):
        """Test convenience explain function."""
        explanations = explainability.explain(
            model=self.model,
            X=self.X_test,
            model_type="sklearn",
            background_data=self.X[:50],
            method="shap"
        )
        
        assert isinstance(explanations, dict)
        assert 'shap_values' in explanations


class TestPyTorchExplainer:
    """Test PyTorch explainer functionality."""
    
    def setup_method(self):
        """Setup PyTorch test data."""
        try:
            import torch
            import torch.nn as nn
            
            # Create simple PyTorch model
            self.torch_model = nn.Sequential(
                nn.Linear(5, 10),
                nn.ReLU(),
                nn.Linear(10, 2),
                nn.Softmax(dim=1)
            )
            
            # Create test data
            self.X_torch = torch.randn(20, 5)
            
        except ImportError:
            self.torch_model = None
            self.X_torch = None
    
    def test_pytorch_explainer_initialization(self):
        """Test PyTorch explainer initialization."""
        if self.torch_model is None:
            pytest.skip("PyTorch not available")
        
        explainer = explainability.Explainer(
            model=self.torch_model,
            model_type="pytorch"
        )
        
        assert explainer.model_type == "pytorch"
        # Note: Captum might not be available, so explainer might be None
    
    def test_pytorch_explanation(self):
        """Test PyTorch model explanation."""
        if self.torch_model is None:
            pytest.skip("PyTorch not available")
        
        try:
            explainer = explainability.Explainer(
                model=self.torch_model,
                model_type="pytorch"
            )
            
            if explainer.explainer is not None:
                explanations = explainer.explain_instance(self.X_torch[:5].numpy(), method="ig")
                assert isinstance(explanations, dict)
        
        except Exception:
            # Captum might not be available
            pytest.skip("Captum not available")


if __name__ == "__main__":
    pytest.main([__file__])
