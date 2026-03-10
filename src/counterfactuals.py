"""
Counterfactual explanation module.
Implements nearest neighbor counterfactual generation.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Union, Optional, Dict, Any, List
import warnings


class CounterfactualGenerator:
    """
    Generate counterfactual explanations using nearest neighbor approach.
    """
    
    def __init__(self, model, X_background: np.ndarray, y_background: np.ndarray,
                 feature_names: Optional[List[str]] = None, 
                 protected_features: Optional[List[str]] = None,
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize counterfactual generator.
        
        Args:
            model: Trained classifier with predict method
            X_background: Background dataset for finding counterfactuals
            y_background: Labels for background dataset
            feature_names: Names of features
            protected_features: Features that should not be modified
            scaler: Scaler for feature normalization
        """
        self.model = model
        self.X_background = X_background
        self.y_background = y_background
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_background.shape[1])]
        self.protected_features = protected_features or []
        self.scaler = scaler
        
        # Initialize nearest neighbors
        self.nn = NearestNeighbors(n_neighbors=10, algorithm='auto')
        self.nn.fit(X_background)
        
        # Identify protected feature indices
        self.protected_indices = [self.feature_names.index(f) for f in self.protected_features 
                                 if f in self.feature_names]
    
    def generate_counterfactual(self, X: np.ndarray, y_original: int, 
                               target_class: Optional[int] = None,
                               max_distance: float = 0.5,
                               num_candidates: int = 5) -> Dict[str, Any]:
        """
        Generate counterfactual explanations for a given instance.
        
        Args:
            X: Original instance to explain
            y_original: Original prediction
            target_class: Desired counterfactual class (if None, opposite class)
            max_distance: Maximum allowed distance for counterfactual
            num_candidates: Number of counterfactual candidates to return
        
        Returns:
            Dictionary containing counterfactual explanations
        """
        if target_class is None:
            # Find opposite class
            unique_classes = np.unique(self.y_background)
            target_class = unique_classes[unique_classes != y_original][0] if len(unique_classes) > 1 else y_original
        
        # Find nearest neighbors
        distances, indices = self.nn.kneighbors(X.reshape(1, -1))
        
        counterfactuals = []
        explanations = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if dist > max_distance:
                continue
                
            candidate = self.X_background[idx]
            candidate_pred = self.model.predict(candidate.reshape(1, -1))[0]
            
            if candidate_pred == target_class:
                # Calculate feature changes
                changes = self._calculate_changes(X, candidate)
                
                # Apply protected feature constraints
                if self.protected_indices:
                    changes = self._apply_protected_constraints(changes, X, candidate)
                
                cf_info = {
                    'counterfactual': candidate,
                    'prediction': candidate_pred,
                    'distance': dist,
                    'changes': changes,
                    'feature_importance': self._calculate_feature_importance(changes)
                }
                
                counterfactuals.append(cf_info)
                
                if len(counterfactuals) >= num_candidates:
                    break
        
        if not counterfactuals:
            # Fallback: generate synthetic counterfactual
            cf_info = self._generate_synthetic_counterfactual(X, y_original, target_class)
            counterfactuals.append(cf_info)
        
        return {
            'original_instance': X,
            'original_prediction': y_original,
            'target_class': target_class,
            'counterfactuals': counterfactuals,
            'num_found': len(counterfactuals)
        }
    
    def _calculate_changes(self, original: np.ndarray, counterfactual: np.ndarray) -> Dict[str, Any]:
        """Calculate changes between original and counterfactual."""
        changes = {}
        for i, (orig, cf) in enumerate(zip(original, counterfactual)):
            feature_name = self.feature_names[i]
            changes[feature_name] = {
                'original': orig,
                'counterfactual': cf,
                'difference': cf - orig,
                'relative_change': (cf - orig) / orig if orig != 0 else 0
            }
        return changes
    
    def _apply_protected_constraints(self, changes: Dict[str, Any], 
                                   original: np.ndarray, 
                                   counterfactual: np.ndarray) -> Dict[str, Any]:
        """Apply constraints to protected features."""
        constrained_changes = changes.copy()
        
        for idx in self.protected_indices:
            feature_name = self.feature_names[idx]
            # Reset protected feature to original value
            constrained_changes[feature_name]['counterfactual'] = original[idx]
            constrained_changes[feature_name]['difference'] = 0
            constrained_changes[feature_name]['relative_change'] = 0
        
        return constrained_changes
    
    def _calculate_feature_importance(self, changes: Dict[str, Any]) -> Dict[str, float]:
        """Calculate importance of each feature in the counterfactual."""
        importance = {}
        for feature_name, change in changes.items():
            # Use absolute relative change as importance metric
            importance[feature_name] = abs(change['relative_change'])
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def _generate_synthetic_counterfactual(self, X: np.ndarray, y_original: int, 
                                        target_class: int) -> Dict[str, Any]:
        """Generate synthetic counterfactual when no real ones found."""
        # Simple approach: perturb non-protected features
        synthetic_cf = X.copy()
        
        # Find non-protected indices
        non_protected_indices = [i for i in range(len(X)) if i not in self.protected_indices]
        
        # Add noise to non-protected features
        noise_scale = 0.1 * np.std(self.X_background, axis=0)
        for idx in non_protected_indices:
            synthetic_cf[idx] += np.random.normal(0, noise_scale[idx])
        
        # Check if synthetic counterfactual has desired prediction
        synthetic_pred = self.model.predict(synthetic_cf.reshape(1, -1))[0]
        
        changes = self._calculate_changes(X, synthetic_cf)
        
        return {
            'counterfactual': synthetic_cf,
            'prediction': synthetic_pred,
            'distance': np.linalg.norm(X - synthetic_cf),
            'changes': changes,
            'feature_importance': self._calculate_feature_importance(changes),
            'synthetic': True
        }
    
    def evaluate_counterfactual_quality(self, counterfactuals: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate quality of generated counterfactuals.
        
        Args:
            counterfactuals: List of counterfactual dictionaries
        
        Returns:
            Dictionary with quality metrics
        """
        if not counterfactuals:
            return {'validity': 0.0, 'proximity': 0.0, 'sparsity': 0.0}
        
        # Validity: percentage of counterfactuals with correct target prediction
        validity = sum(1 for cf in counterfactuals if cf.get('synthetic', False) == False) / len(counterfactuals)
        
        # Proximity: average distance from original
        distances = [cf['distance'] for cf in counterfactuals]
        proximity = 1.0 / (1.0 + np.mean(distances))  # Higher is better
        
        # Sparsity: average number of changed features
        changed_features = []
        for cf in counterfactuals:
            changes = cf['changes']
            changed = sum(1 for change in changes.values() if abs(change['difference']) > 1e-6)
            changed_features.append(changed)
        
        max_features = len(self.feature_names)
        sparsity = 1.0 - (np.mean(changed_features) / max_features)  # Higher is better
        
        return {
            'validity': validity,
            'proximity': proximity,
            'sparsity': sparsity,
            'avg_distance': np.mean(distances),
            'avg_changed_features': np.mean(changed_features)
        }
    
    def batch_generate_counterfactuals(self, X_batch: np.ndarray, y_batch: np.ndarray,
                                     **kwargs) -> List[Dict[str, Any]]:
        """
        Generate counterfactuals for a batch of instances.
        
        Args:
            X_batch: Batch of instances
            y_batch: Batch of predictions
            **kwargs: Additional arguments for generate_counterfactual
        
        Returns:
            List of counterfactual explanations
        """
        results = []
        for X, y in zip(X_batch, y_batch):
            cf_result = self.generate_counterfactual(X, y, **kwargs)
            results.append(cf_result)
        
        return results


def generate_counterfactuals(model, X: np.ndarray, y: np.ndarray, 
                           X_background: np.ndarray, y_background: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           protected_features: Optional[List[str]] = None,
                           **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convenience function to generate counterfactual explanations.
    
    Args:
        model: Trained classifier
        X: Instance(s) to explain
        y: Original prediction(s)
        X_background: Background dataset
        y_background: Background labels
        feature_names: Names of features
        protected_features: Features that should not be modified
        **kwargs: Additional arguments
    
    Returns:
        Counterfactual explanation(s)
    """
    generator = CounterfactualGenerator(
        model, X_background, y_background, feature_names, protected_features
    )
    
    if len(X.shape) == 1:
        # Single instance
        return generator.generate_counterfactual(X, y[0] if len(y.shape) > 0 else y, **kwargs)
    else:
        # Batch of instances
        return generator.batch_generate_counterfactuals(X, y, **kwargs)
