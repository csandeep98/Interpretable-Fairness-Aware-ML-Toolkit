"""
Fairness metrics and evaluation module.
Implements group fairness metrics and subgroup analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from typing import Union, Optional, Dict, Any, List
import warnings


def statistical_parity_difference(y_true: np.ndarray, y_pred: np.ndarray, 
                                sensitive_attr: np.ndarray,
                                privileged_group: Union[int, str] = 1) -> float:
    """
    Calculate statistical parity difference.
    
    Args:
        y_true: True labels (not used in calculation, included for consistency)
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values (0/1 or categorical)
        privileged_group: Value indicating privileged group
    
    Returns:
        Statistical parity difference (privileged - unprivileged)
    """
    groups = np.unique(sensitive_attr)
    rates = {}
    
    for g in groups:
        mask = (sensitive_attr == g)
        if np.sum(mask) == 0:
            rates[g] = np.nan
        else:
            rates[g] = np.mean(y_pred[mask])
    
    if privileged_group not in rates:
        raise ValueError(f"Privileged group {privileged_group} not found in sensitive attribute")
    
    # Find unprivileged group (assuming binary)
    unprivileged_groups = [g for g in groups if g != privileged_group]
    if not unprivileged_groups:
        warnings.warn("Only one group found in sensitive attribute")
        return 0.0
    
    unprivileged_group = unprivileged_groups[0]
    return rates[privileged_group] - rates[unprivileged_group]


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, 
                               sensitive_attr: np.ndarray,
                               privileged_group: Union[int, str] = 1,
                               positive_label: int = 1) -> float:
    """
    Calculate equal opportunity difference (true positive rate difference).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        privileged_group: Value indicating privileged group
        positive_label: Positive class label
    
    Returns:
        Equal opportunity difference (privileged - unprivileged)
    """
    groups = np.unique(sensitive_attr)
    tpr_rates = {}
    
    for g in groups:
        mask = (sensitive_attr == g)
        if np.sum(mask) == 0:
            tpr_rates[g] = np.nan
            continue
            
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        tpr_rates[g] = tpr
    
    if privileged_group not in tpr_rates:
        raise ValueError(f"Privileged group {privileged_group} not found in sensitive attribute")
    
    unprivileged_groups = [g for g in groups if g != privileged_group]
    if not unprivileged_groups:
        warnings.warn("Only one group found in sensitive attribute")
        return 0.0
    
    unprivileged_group = unprivileged_groups[0]
    return tpr_rates[privileged_group] - tpr_rates[unprivileged_group]


def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, 
                  sensitive_attr: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate equalized odds metrics (TPR and FPR for each group).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
    
    Returns:
        Dictionary with TPR and FPR for each group
    """
    groups = np.unique(sensitive_attr)
    odds = {}
    
    for g in groups:
        mask = (sensitive_attr == g)
        if np.sum(mask) == 0:
            odds[g] = {'tpr': np.nan, 'fpr': np.nan}
            continue
            
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        
        odds[g] = {'tpr': tpr, 'fpr': fpr}
    
    return odds


def disparate_impact_ratio(y_pred: np.ndarray, sensitive_attr: np.ndarray,
                         privileged_group: Union[int, str] = 1) -> float:
    """
    Calculate disparate impact ratio.
    
    Args:
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        privileged_group: Value indicating privileged group
    
    Returns:
        Disparate impact ratio (unprivileged / privileged)
    """
    groups = np.unique(sensitive_attr)
    rates = {}
    
    for g in groups:
        mask = (sensitive_attr == g)
        if np.sum(mask) == 0:
            rates[g] = np.nan
        else:
            rates[g] = np.mean(y_pred[mask])
    
    if privileged_group not in rates:
        raise ValueError(f"Privileged group {privileged_group} not found in sensitive attribute")
    
    unprivileged_groups = [g for g in groups if g != privileged_group]
    if not unprivileged_groups:
        warnings.warn("Only one group found in sensitive attribute")
        return 1.0
    
    unprivileged_group = unprivileged_groups[0]
    
    # Avoid division by zero
    if rates[privileged_group] == 0:
        return np.inf if rates[unprivileged_group] > 0 else 1.0
    
    return rates[unprivileged_group] / rates[privileged_group]


def accuracy_difference(y_true: np.ndarray, y_pred: np.ndarray, 
                       sensitive_attr: np.ndarray,
                       privileged_group: Union[int, str] = 1) -> float:
    """
    Calculate accuracy difference between groups.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        privileged_group: Value indicating privileged group
    
    Returns:
        Accuracy difference (privileged - unprivileged)
    """
    groups = np.unique(sensitive_attr)
    acc_rates = {}
    
    for g in groups:
        mask = (sensitive_attr == g)
        if np.sum(mask) == 0:
            acc_rates[g] = np.nan
        else:
            acc_rates[g] = accuracy_score(y_true[mask], y_pred[mask])
    
    if privileged_group not in acc_rates:
        raise ValueError(f"Privileged group {privileged_group} not found in sensitive attribute")
    
    unprivileged_groups = [g for g in groups if g != privileged_group]
    if not unprivileged_groups:
        warnings.warn("Only one group found in sensitive attribute")
        return 0.0
    
    unprivileged_group = unprivileged_groups[0]
    return acc_rates[privileged_group] - acc_rates[unprivileged_group]


def subgroup_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            sensitive_attrs: Dict[str, np.ndarray],
                            privileged_groups: Optional[Dict[str, Union[int, str]]] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate fairness metrics for multiple sensitive attributes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attrs: Dictionary of sensitive attribute name -> values
        privileged_groups: Dictionary of privileged group values for each attribute
    
    Returns:
        Dictionary with fairness metrics for each sensitive attribute
    """
    if privileged_groups is None:
        privileged_groups = {}
    
    results = {}
    
    for attr_name, attr_values in sensitive_attrs.items():
        privileged_group = privileged_groups.get(attr_name, 1)
        
        try:
            metrics = {
                'statistical_parity_difference': statistical_parity_difference(
                    y_true, y_pred, attr_values, privileged_group
                ),
                'equal_opportunity_difference': equal_opportunity_difference(
                    y_true, y_pred, attr_values, privileged_group
                ),
                'disparate_impact_ratio': disparate_impact_ratio(
                    y_pred, attr_values, privileged_group
                ),
                'accuracy_difference': accuracy_difference(
                    y_true, y_pred, attr_values, privileged_group
                )
            }
            
            # Add equalized odds details
            odds = equalized_odds(y_true, y_pred, attr_values)
            metrics['equalized_odds'] = odds
            
            results[attr_name] = metrics
            
        except Exception as e:
            warnings.warn(f"Error calculating metrics for {attr_name}: {str(e)}")
            results[attr_name] = {'error': str(e)}
    
    return results


class FairnessAuditor:
    """
    Comprehensive fairness auditing tool.
    """
    
    def __init__(self, sensitive_attrs: Dict[str, np.ndarray],
                 privileged_groups: Optional[Dict[str, Union[int, str]]] = None,
                 attribute_names: Optional[Dict[str, str]] = None):
        """
        Initialize fairness auditor.
        
        Args:
            sensitive_attrs: Dictionary of sensitive attribute name -> values
            privileged_groups: Dictionary of privileged group values
            attribute_names: Human-readable names for attributes
        """
        self.sensitive_attrs = sensitive_attrs
        self.privileged_groups = privileged_groups or {}
        self.attribute_names = attribute_names or {}
        self.results = {}
    
    def audit(self, y_true: np.ndarray, y_pred: np.ndarray, 
              y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive fairness audit.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
        
        Returns:
            Comprehensive audit results
        """
        audit_results = {
            'overall_metrics': {
                'accuracy': accuracy_score(y_true, y_pred),
                'num_samples': len(y_true),
                'positive_rate': np.mean(y_pred)
            },
            'fairness_metrics': subgroup_fairness_metrics(
                y_true, y_pred, self.sensitive_attrs, self.privileged_groups
            )
        }
        
        # Add probability-based metrics if available
        if y_prob is not None:
            audit_results['probability_metrics'] = self._calculate_probability_metrics(
                y_true, y_prob
            )
        
        # Add subgroup analysis
        audit_results['subgroup_analysis'] = self._subgroup_analysis(y_true, y_pred)
        
        # Add fairness summary
        audit_results['fairness_summary'] = self._generate_fairness_summary(
            audit_results['fairness_metrics']
        )
        
        self.results = audit_results
        return audit_results
    
    def _calculate_probability_metrics(self, y_true: np.ndarray, 
                                     y_prob: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate probability-based fairness metrics."""
        metrics = {}
        
        for attr_name, attr_values in self.sensitive_attrs.items():
            try:
                groups = np.unique(attr_values)
                group_metrics = {}
                
                for g in groups:
                    mask = (attr_values == g)
                    if np.sum(mask) == 0:
                        continue
                    
                    y_true_g = y_true[mask]
                    y_prob_g = y_prob[mask]
                    
                    if len(np.unique(y_true_g)) > 1:
                        group_metrics[f'group_{g}_auc'] = roc_auc_score(y_true_g, y_prob_g)
                    else:
                        group_metrics[f'group_{g}_auc'] = np.nan
                    
                    group_metrics[f'group_{g}_avg_prob'] = np.mean(y_prob_g)
                
                metrics[attr_name] = group_metrics
                
            except Exception as e:
                warnings.warn(f"Error calculating probability metrics for {attr_name}: {str(e)}")
                metrics[attr_name] = {'error': str(e)}
        
        return metrics
    
    def _subgroup_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform detailed subgroup analysis."""
        analysis = {}
        
        for attr_name, attr_values in self.sensitive_attrs.items():
            groups = np.unique(attr_values)
            group_stats = {}
            
            for g in groups:
                mask = (attr_values == g)
                if np.sum(mask) == 0:
                    continue
                
                y_true_g = y_true[mask]
                y_pred_g = y_pred[mask]
                
                tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()
                
                group_stats[f'group_{g}'] = {
                    'size': np.sum(mask),
                    'accuracy': accuracy_score(y_true_g, y_pred_g),
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
                    'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan,
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn)
                }
            
            analysis[attr_name] = group_stats
        
        return analysis
    
    def _generate_fairness_summary(self, fairness_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate summary of fairness assessment."""
        summary = {
            'overall_fairness_score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'passed_checks': []
        }
        
        total_score = 0
        num_metrics = 0
        
        for attr_name, metrics in fairness_metrics.items():
            if 'error' in metrics:
                continue
            
            # Statistical parity (ideal: 0, threshold: ±0.1)
            spd = metrics.get('statistical_parity_difference', 0)
            if abs(spd) > 0.2:
                summary['critical_issues'].append(f"{attr_name}: High statistical parity difference ({spd:.3f})")
            elif abs(spd) > 0.1:
                summary['warnings'].append(f"{attr_name}: Moderate statistical parity difference ({spd:.3f})")
            else:
                summary['passed_checks'].append(f"{attr_name}: Statistical parity within acceptable range")
            
            # Equal opportunity (ideal: 0, threshold: ±0.1)
            eod = metrics.get('equal_opportunity_difference', 0)
            if abs(eod) > 0.2:
                summary['critical_issues'].append(f"{attr_name}: High equal opportunity difference ({eod:.3f})")
            elif abs(eod) > 0.1:
                summary['warnings'].append(f"{attr_name}: Moderate equal opportunity difference ({eod:.3f})")
            else:
                summary['passed_checks'].append(f"{attr_name}: Equal opportunity within acceptable range")
            
            # Disparate impact (ideal: 1.0, range: 0.8-1.25)
            dir_ratio = metrics.get('disparate_impact_ratio', 1.0)
            if dir_ratio < 0.5 or dir_ratio > 2.0:
                summary['critical_issues'].append(f"{attr_name}: Extreme disparate impact ({dir_ratio:.3f})")
            elif dir_ratio < 0.8 or dir_ratio > 1.25:
                summary['warnings'].append(f"{attr_name}: Moderate disparate impact ({dir_ratio:.3f})")
            else:
                summary['passed_checks'].append(f"{attr_name}: Disparate impact within legal range")
            
            # Calculate fairness score
            fairness_score = max(0, 1 - (abs(spd) + abs(eod)) / 2)
            total_score += fairness_score
            num_metrics += 1
        
        if num_metrics > 0:
            summary['overall_fairness_score'] = total_score / num_metrics
        
        return summary
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on audit results."""
        if not self.results:
            return ["Run audit() first to generate recommendations"]
        
        recommendations = []
        summary = self.results.get('fairness_summary', {})
        
        if summary.get('overall_fairness_score', 0) < 0.7:
            recommendations.append("Consider implementing fairness-aware preprocessing techniques")
        
        critical_issues = summary.get('critical_issues', [])
        if critical_issues:
            recommendations.append("Address critical fairness issues before deployment")
            recommendations.append("Consider using fairness constraints in model training")
        
        warnings_list = summary.get('warnings', [])
        if warnings_list:
            recommendations.append("Monitor fairness metrics regularly for potential drift")
            recommendations.append("Consider implementing post-processing fairness adjustments")
        
        if not critical_issues and not warnings_list:
            recommendations.append("Model appears fair - continue monitoring in production")
        
        return recommendations
