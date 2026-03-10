"""
Test suite for fairness module.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import fairness


class TestFairnessMetrics:
    """Test fairness metrics functions."""
    
    def setup_method(self):
        """Setup test data."""
        # Create test data
        np.random.seed(42)
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.sensitive_attr = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    def test_statistical_parity_difference(self):
        """Test statistical parity difference calculation."""
        spd = fairness.statistical_parity_difference(
            self.y_true, self.y_pred, self.sensitive_attr
        )
        
        assert isinstance(spd, float)
        # Group 0: predictions [0,1,0,0,0] -> mean = 0.2
        # Group 1: predictions [1,1,0,1,1] -> mean = 0.8
        # SPD = 0.8 - 0.2 = 0.6
        assert abs(spd - 0.6) < 1e-10
    
    def test_equal_opportunity_difference(self):
        """Test equal opportunity difference calculation."""
        eod = fairness.equal_opportunity_difference(
            self.y_true, self.y_pred, self.sensitive_attr
        )
        
        assert isinstance(eod, float)
        # TPR Group 0: TP=1, FN=1 -> TPR=0.5
        # TPR Group 1: TP=3, FN=1 -> TPR=0.75
        # EOD = 0.75 - 0.5 = 0.25
        assert abs(eod - 0.25) < 1e-10
    
    def test_equalized_odds(self):
        """Test equalized odds calculation."""
        odds = fairness.equalized_odds(self.y_true, self.y_pred, self.sensitive_attr)
        
        assert isinstance(odds, dict)
        assert 0 in odds
        assert 1 in odds
        assert 'tpr' in odds[0]
        assert 'fpr' in odds[0]
        assert 'tpr' in odds[1]
        assert 'fpr' in odds[1]
    
    def test_disparate_impact_ratio(self):
        """Test disparate impact ratio calculation."""
        dir_ratio = fairness.disparate_impact_ratio(
            self.y_pred, self.sensitive_attr
        )
        
        assert isinstance(dir_ratio, float)
        # Group 0 rate = 0.2, Group 1 rate = 0.8
        # DIR = 0.2 / 0.8 = 0.25
        assert abs(dir_ratio - 0.25) < 1e-10
    
    def test_accuracy_difference(self):
        """Test accuracy difference calculation."""
        acc_diff = fairness.accuracy_difference(
            self.y_true, self.y_pred, self.sensitive_attr
        )
        
        assert isinstance(acc_diff, float)
        # Group 0: [0,1,0,0,0] vs [0,1,0,1,0] -> accuracy = 4/5 = 0.8
        # Group 1: [1,1,0,1,1] vs [0,1,0,1,1] -> accuracy = 4/5 = 0.8
        # Diff = 0.8 - 0.8 = 0.0
        assert abs(acc_diff - 0.0) < 1e-10


class TestFairnessAuditor:
    """Test FairnessAuditor class."""
    
    def setup_method(self):
        """Setup test data and auditor."""
        # Create synthetic dataset
        X, y = make_classification(
            n_samples=200, n_features=5, n_informative=3, 
            n_redundant=2, random_state=42
        )
        
        # Create sensitive attribute with bias
        np.random.seed(42)
        sensitive_attr = np.random.choice([0, 1], size=len(y), p=[0.6, 0.4])
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        # Make predictions
        self.y_pred = self.model.predict(X)
        self.y_prob = self.model.predict_proba(X)[:, 1]
        
        # Initialize auditor
        self.auditor = fairness.FairnessAuditor(
            sensitive_attrs={'sensitive_attr': sensitive_attr},
            privileged_groups={'sensitive_attr': 1}
        )
        
        self.y_true = y
        self.sensitive_attr = sensitive_attr
    
    def test_audit_initialization(self):
        """Test auditor initialization."""
        assert 'sensitive_attr' in self.auditor.sensitive_attrs
        assert self.auditor.privileged_groups['sensitive_attr'] == 1
    
    def test_audit_execution(self):
        """Test audit execution."""
        results = self.auditor.audit(self.y_true, self.y_pred, self.y_prob)
        
        assert 'overall_metrics' in results
        assert 'fairness_metrics' in results
        assert 'fairness_summary' in results
        
        # Check overall metrics
        overall = results['overall_metrics']
        assert 'accuracy' in overall
        assert 'num_samples' in overall
        assert overall['num_samples'] == len(self.y_true)
    
    def test_fairness_summary_generation(self):
        """Test fairness summary generation."""
        results = self.auditor.audit(self.y_true, self.y_pred, self.y_prob)
        summary = results['fairness_summary']
        
        assert 'overall_fairness_score' in summary
        assert isinstance(summary['overall_fairness_score'], float)
        assert 0 <= summary['overall_fairness_score'] <= 1
    
    def test_recommendations_generation(self):
        """Test recommendations generation."""
        # Run audit first
        self.auditor.audit(self.y_true, self.y_pred, self.y_prob)
        
        # Generate recommendations
        recommendations = self.auditor.generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__])
