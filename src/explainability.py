"""
Explainability module for interpretable ML models.
Provides wrappers for SHAP and Captum explainers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Union, Optional, Dict, Any
import warnings

try:
    import torch
    import torch.nn as nn
    from captum.attr import IntegratedGradients, DeepLift, GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    warnings.warn("Captum not available. PyTorch model explanations will be disabled.")


class Explainer:
    """
    Unified explainer interface supporting SHAP and Captum backends.
    """
    
    def __init__(self, model, model_type: str = "sklearn", background_data: Optional[np.ndarray] = None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (sklearn, PyTorch, etc.)
            model_type: Type of model ("sklearn", "pytorch")
            background_data: Background data for SHAP (required for some explainers)
        """
        self.model = model
        self.model_type = model_type.lower()
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate explainer based on model type."""
        if self.model_type == "sklearn":
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.TreeExplainer(self.model, data=self.background_data)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
        
        elif self.model_type == "pytorch" and CAPTUM_AVAILABLE:
            if isinstance(self.model, nn.Module):
                self.explainer = IntegratedGradients(self.model)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def explain_instance(self, X: np.ndarray, method: str = "shap") -> Dict[str, Any]:
        """
        Explain a single instance or batch of instances.
        
        Args:
            X: Input data to explain
            method: Explanation method ("shap", "ig", "deeplift")
        
        Returns:
            Dictionary containing explanations
        """
        if method == "shap":
            return self._explain_shap(X)
        elif method in ["ig", "deeplift", "gradient_shap"] and self.model_type == "pytorch":
            return self._explain_captum(X, method)
        else:
            raise ValueError(f"Method {method} not supported for model type {self.model_type}")
    
    def _explain_shap(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP explanations."""
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # For multi-class, return explanations for each class
            explanations = {}
            for i, values in enumerate(shap_values):
                explanations[f"class_{i}"] = {
                    'shap_values': values,
                    'base_values': self.explainer.expected_value[i] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                    'data': X
                }
            return explanations
        else:
            return {
                'shap_values': shap_values,
                'base_values': self.explainer.expected_value,
                'data': X
            }
    
    def _explain_captum(self, X: np.ndarray, method: str) -> Dict[str, Any]:
        """Generate Captum explanations for PyTorch models."""
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is not available")
        
        if self.explainer is None:
            raise ValueError("Captum explainer not initialized")
        
        # Convert numpy to tensor
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        else:
            X_tensor = X
        
        # Initialize appropriate explainer
        if method == "ig":
            explainer = IntegratedGradients(self.model)
        elif method == "deeplift":
            explainer = DeepLift(self.model)
        elif method == "gradient_shap":
            explainer = GradientShap(self.model)
        else:
            raise ValueError(f"Unsupported Captum method: {method}")
        
        # Generate attributions
        if method == "gradient_shap" and self.background_data is not None:
            baseline = torch.tensor(self.background_data[:1], dtype=torch.float32)
            attributions = explainer.attribute(X_tensor, baselines=baseline)
        else:
            attributions = explainer.attribute(X_tensor)
        
        return {
            'attributions': attributions.detach().numpy(),
            'data': X,
            'method': method
        }
    
    def plot_summary(self, explanations: Dict[str, Any], feature_names: Optional[list] = None, 
                     class_names: Optional[list] = None, max_display: int = 20):
        """
        Create summary plots for explanations.
        
        Args:
            explanations: Dictionary from explain_instance
            feature_names: Names of features
            class_names: Names of classes (for multi-class)
            max_display: Maximum number of features to display
        """
        if 'shap_values' in explanations:
            # Single output case
            shap.summary_plot(
                explanations['shap_values'], 
                explanations['data'],
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
        elif 'class_0' in explanations:
            # Multi-class case
            for class_name, exp in explanations.items():
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    exp['shap_values'],
                    exp['data'],
                    feature_names=feature_names,
                    class_names=[class_name],
                    max_display=max_display,
                    show=False
                )
                plt.title(f"SHAP Summary - {class_name}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_waterfall(self, explanations: Dict[str, Any], instance_idx: int = 0, 
                      feature_names: Optional[list] = None, max_display: int = 20):
        """
        Create waterfall plot for a single instance.
        
        Args:
            explanations: Dictionary from explain_instance
            instance_idx: Index of instance to plot
            feature_names: Names of features
            max_display: Maximum number of features to display
        """
        if 'shap_values' in explanations:
            shap.waterfall_plot(
                shap.Explanation(
                    values=explanations['shap_values'][instance_idx],
                    base_values=explanations['base_values'],
                    data=explanations['data'][instance_idx],
                    feature_names=feature_names
                ),
                max_display=max_display,
                show=False
            )
        plt.show()


def explain(model, X: np.ndarray, model_type: str = "sklearn", 
           background_data: Optional[np.ndarray] = None, method: str = "shap") -> Dict[str, Any]:
    """
    Convenience function to explain model predictions.
    
    Args:
        model: Trained model
        X: Input data to explain
        model_type: Type of model ("sklearn", "pytorch")
        background_data: Background data for SHAP
        method: Explanation method
    
    Returns:
        Dictionary containing explanations
    """
    explainer = Explainer(model, model_type, background_data)
    return explainer.explain_instance(X, method)
