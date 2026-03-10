"""
Report generation module for ML model audits.
Creates comprehensive HTML audit reports with visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
import json
import warnings

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AuditReportGenerator:
    """
    Generate comprehensive HTML audit reports for ML models.
    """
    
    def __init__(self, model_info: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.
        
        Args:
            model_info: Dictionary with model metadata
        """
        self.model_info = model_info or {}
        self.report_data = {}
        
    def generate_report(self, fairness_results: Dict[str, Any],
                       explainability_results: Optional[Dict[str, Any]] = None,
                       counterfactual_results: Optional[Dict[str, Any]] = None,
                       output_path: str = "audit_report.html") -> str:
        """
        Generate comprehensive audit report.
        
        Args:
            fairness_results: Results from fairness audit
            explainability_results: Results from explainability analysis
            counterfactual_results: Results from counterfactual analysis
            output_path: Path to save HTML report
        
        Returns:
            Path to generated report
        """
        self.report_data = {
            'fairness': fairness_results,
            'explainability': explainability_results,
            'counterfactuals': counterfactual_results,
            'timestamp': datetime.now().isoformat(),
            'model_info': self.model_info
        }
        
        html_content = self._generate_html()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_html(self) -> str:
        """Generate complete HTML report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Model Fairness Audit Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_executive_summary()}
        {self._generate_fairness_section()}
        {self._generate_explainability_section()}
        {self._generate_counterfactuals_section()}
        {self._generate_recommendations()}
        {self._generate_footer()}
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
        """
        return html
    
    def _get_css_styles(self) -> str:
        """Return CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 3px solid #007bff;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #007bff;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            color: #666;
            font-size: 1.2em;
        }
        
        .section {
            margin: 40px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }
        
        .section h2 {
            color: #007bff;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #007bff;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #007bff;
        }
        
        .metric-card.warning {
            border-left-color: #ffc107;
        }
        
        .metric-card.danger {
            border-left-color: #dc3545;
        }
        
        .metric-card.success {
            border-left-color: #28a745;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        
        .alert {
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        .alert-danger {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .table th, .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .table th {
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
        
        .table tr:hover {
            background-color: #f5f5f5;
        }
        
        .footer {
            margin-top: 50px;
            padding: 20px 0;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }
        
        .collapsible {
            background-color: #f1f1f1;
            color: #444;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            margin: 10px 0;
        }
        
        .active, .collapsible:hover {
            background-color: #ccc;
        }
        
        .content {
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background-color: #f1f1f1;
        }
        """
    
    def _generate_header(self) -> str:
        """Generate report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = self.model_info.get('name', 'Unknown Model')
        
        return f"""
        <div class="header">
            <h1>ML Model Fairness Audit Report</h1>
            <div class="subtitle">
                Model: {model_name}<br>
                Generated: {timestamp}
            </div>
        </div>
        """
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        fairness_data = self.report_data.get('fairness', {})
        summary = fairness_data.get('fairness_summary', {})
        overall_score = summary.get('overall_fairness_score', 0)
        
        # Determine status
        if overall_score >= 0.8:
            status = "success"
            status_text = "PASS"
            status_desc = "Model demonstrates good fairness characteristics"
        elif overall_score >= 0.6:
            status = "warning"
            status_text = "WARNING"
            status_desc = "Model has some fairness concerns that should be addressed"
        else:
            status = "danger"
            status_text = "FAIL"
            status_desc = "Model has significant fairness issues"
        
        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card {status}">
                    <div class="metric-value">{status_text}</div>
                    <div class="metric-label">Overall Fairness Status</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{overall_score:.2f}</div>
                    <div class="metric-label">Fairness Score (0-1)</div>
                </div>
            </div>
            <p><strong>Assessment:</strong> {status_desc}</p>
        </div>
        """
    
    def _generate_fairness_section(self) -> str:
        """Generate fairness analysis section."""
        fairness_data = self.report_data.get('fairness', {})
        
        if not fairness_data:
            return '<div class="section"><h2>Fairness Analysis</h2><p>No fairness data available.</p></div>'
        
        html = '<div class="section"><h2>Fairness Analysis</h2>'
        
        # Overall metrics
        overall_metrics = fairness_data.get('overall_metrics', {})
        if overall_metrics:
            html += '<h3>Overall Model Performance</h3>'
            html += '<div class="metric-grid">'
            for metric, value in overall_metrics.items():
                if isinstance(value, float):
                    display_value = f"{value:.3f}"
                else:
                    display_value = str(value)
                html += f'''
                <div class="metric-card">
                    <div class="metric-value">{display_value}</div>
                    <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                </div>
                '''
            html += '</div>'
        
        # Fairness metrics by attribute
        fairness_metrics = fairness_data.get('fairness_metrics', {})
        if fairness_metrics:
            html += '<h3>Fairness Metrics by Protected Attribute</h3>'
            
            for attr_name, metrics in fairness_metrics.items():
                if 'error' in metrics:
                    html += f'<div class="alert alert-warning">Error calculating metrics for {attr_name}: {metrics["error"]}</div>'
                    continue
                
                html += f'<h4>{attr_name.replace("_", " ").title()}</h4>'
                html += '<div class="metric-grid">'
                
                key_metrics = {
                    'statistical_parity_difference': 'Statistical Parity Difference',
                    'equal_opportunity_difference': 'Equal Opportunity Difference',
                    'disparate_impact_ratio': 'Disparate Impact Ratio',
                    'accuracy_difference': 'Accuracy Difference'
                }
                
                for metric_key, metric_label in key_metrics.items():
                    if metric_key in metrics:
                        value = metrics[metric_key]
                        if isinstance(value, float):
                            display_value = f"{value:.3f}"
                        else:
                            display_value = str(value)
                        
                        # Determine card style based on metric
                        card_class = 'metric-card'
                        if metric_key == 'disparate_impact_ratio':
                            if value < 0.8 or value > 1.25:
                                card_class += ' warning'
                            if value < 0.5 or value > 2.0:
                                card_class += ' danger'
                        else:
                            if abs(value) > 0.1:
                                card_class += ' warning'
                            if abs(value) > 0.2:
                                card_class += ' danger'
                        
                        html += f'''
                        <div class="{card_class}">
                            <div class="metric-value">{display_value}</div>
                            <div class="metric-label">{metric_label}</div>
                        </div>
                        '''
                
                html += '</div>'
        
        # Fairness summary
        summary = fairness_data.get('fairness_summary', {})
        if summary:
            html += '<h3>Fairness Assessment Summary</h3>'
            
            if summary.get('critical_issues'):
                html += '<div class="alert alert-danger"><strong>Critical Issues:</strong><ul>'
                for issue in summary['critical_issues']:
                    html += f'<li>{issue}</li>'
                html += '</ul></div>'
            
            if summary.get('warnings'):
                html += '<div class="alert alert-warning"><strong>Warnings:</strong><ul>'
                for warning in summary['warnings']:
                    html += f'<li>{warning}</li>'
                html += '</ul></div>'
            
            if summary.get('passed_checks'):
                html += '<div class="alert alert-success"><strong>Passed Checks:</strong><ul>'
                for check in summary['passed_checks']:
                    html += f'<li>{check}</li>'
                html += '</ul></div>'
        
        html += '</div>'
        return html
    
    def _generate_explainability_section(self) -> str:
        """Generate explainability section."""
        exp_data = self.report_data.get('explainability')
        
        if not exp_data:
            return '<div class="section"><h2>Explainability Analysis</h2><p>No explainability data available.</p></div>'
        
        html = '<div class="section"><h2>Explainability Analysis</h2>'
        
        # Add SHAP summary plot if available
        if 'shap_values' in exp_data:
            html += '''
            <h3>Feature Importance (SHAP)</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{}" alt="SHAP Summary Plot">
            </div>
            '''.format(self._create_shap_summary_plot(exp_data))
        
        html += '<p>Model explainability analysis provides insights into which features are driving predictions.</p>'
        html += '</div>'
        
        return html
    
    def _generate_counterfactuals_section(self) -> str:
        """Generate counterfactuals section."""
        cf_data = self.report_data.get('counterfactuals')
        
        if not cf_data:
            return '<div class="section"><h2>Counterfactual Analysis</h2><p>No counterfactual data available.</p></div>'
        
        html = '<div class="section"><h2>Counterfactual Analysis</h2>'
        
        if isinstance(cf_data, dict) and 'counterfactuals' in cf_data:
            num_found = cf_data.get('num_found', 0)
            html += f'<p>Found {num_found} counterfactual explanations.</p>'
            
            if num_found > 0:
                html += '<h3>Counterfactual Quality Metrics</h3>'
                # Add quality metrics visualization
                html += self._create_counterfactual_quality_plot(cf_data)
        
        html += '</div>'
        return html
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations section."""
        fairness_data = self.report_data.get('fairness', {})
        summary = fairness_data.get('fairness_summary', {})
        
        html = '<div class="section"><h2>Recommendations</h2>'
        
        # Generate recommendations based on fairness summary
        recommendations = []
        
        if summary.get('overall_fairness_score', 0) < 0.7:
            recommendations.append("Consider implementing fairness-aware preprocessing techniques")
        
        if summary.get('critical_issues'):
            recommendations.append("Address critical fairness issues before deployment")
            recommendations.append("Consider using fairness constraints in model training")
        
        if summary.get('warnings'):
            recommendations.append("Monitor fairness metrics regularly for potential drift")
            recommendations.append("Consider implementing post-processing fairness adjustments")
        
        if not recommendations:
            recommendations.append("Model appears fair - continue monitoring in production")
        
        html += '<ul>'
        for rec in recommendations:
            html += f'<li>{rec}</li>'
        html += '</ul>'
        
        html += '</div>'
        return html
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
        <div class="footer">
            <p>Report generated by Interpretable Fairness-Aware ML Toolkit</p>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
    
    def _get_javascript(self) -> str:
        """Return JavaScript for interactive elements."""
        return """
        // Collapsible sections
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }
        """
    
    def _create_shap_summary_plot(self, exp_data: Dict[str, Any]) -> str:
        """Create SHAP summary plot and return as base64."""
        try:
            import shap
            
            # Create a simple matplotlib plot as fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if 'shap_values' in exp_data:
                shap_values = exp_data['shap_values']
                data = exp_data.get('data', np.random.random((100, 5)))
                
                # Create feature importance plot
                if isinstance(shap_values, list):
                    # Multi-class case - use first class
                    importance = np.abs(shap_values[0]).mean(0)
                else:
                    importance = np.abs(shap_values).mean(0)
                
                feature_names = [f"Feature_{i}" for i in range(len(importance))]
                
                y_pos = np.arange(len(feature_names))
                ax.barh(y_pos, importance)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names)
                ax.set_xlabel('Mean |SHAP Value|')
                ax.set_title('Feature Importance (SHAP)')
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            warnings.warn(f"Could not create SHAP plot: {str(e)}")
            return ""
    
    def _create_counterfactual_quality_plot(self, cf_data: Dict[str, Any]) -> str:
        """Create counterfactual quality metrics plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            counterfactuals = cf_data.get('counterfactuals', [])
            if counterfactuals:
                distances = [cf.get('distance', 0) for cf in counterfactuals]
                changes = [len([c for c in cf.get('changes', {}).values() 
                             if abs(c.get('difference', 0)) > 1e-6]) 
                          for cf in counterfactuals]
                
                x = range(len(counterfactuals))
                width = 0.35
                
                ax.bar([i - width/2 for i in x], distances, width, label='Distance', alpha=0.7)
                ax.bar([i + width/2 for i in x], changes, width, label='Changed Features', alpha=0.7)
                
                ax.set_xlabel('Counterfactual Index')
                ax.set_ylabel('Value')
                ax.set_title('Counterfactual Quality Metrics')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'<div class="chart-container"><img src="data:image/png;base64,{image_base64}" alt="Counterfactual Quality Metrics"></div>'
            
        except Exception as e:
            warnings.warn(f"Could not create counterfactual plot: {str(e)}")
            return ""


def generate_audit_report(fairness_results: Dict[str, Any],
                         explainability_results: Optional[Dict[str, Any]] = None,
                         counterfactual_results: Optional[Dict[str, Any]] = None,
                         model_info: Optional[Dict[str, Any]] = None,
                         output_path: str = "audit_report.html") -> str:
    """
    Convenience function to generate audit report.
    
    Args:
        fairness_results: Results from fairness audit
        explainability_results: Results from explainability analysis
        counterfactual_results: Results from counterfactual analysis
        model_info: Model metadata
        output_path: Path to save report
    
    Returns:
        Path to generated report
    """
    generator = AuditReportGenerator(model_info)
    return generator.generate_report(
        fairness_results, explainability_results, counterfactual_results, output_path
    )
