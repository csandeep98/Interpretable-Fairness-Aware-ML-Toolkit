# Interpretable Fairness-Aware ML Toolkit

A comprehensive open-source toolkit for building interpretable and fair machine learning models. This toolkit integrates explainability, counterfactuals, subgroup fairness metrics, and automated audit reports to provide audit-ready tools for policy and compliance teams.

## 🌟 Why This Toolkit Matters

**Policy & Compliance Ready**: Organizations need audit-ready tools to ensure their ML models meet regulatory requirements and ethical standards. This toolkit provides comprehensive fairness analysis and explainability features that are essential for modern AI governance.

**Comprehensive Integration**: Unlike fragmented solutions, this toolkit combines:
- **Explainability** (SHAP & Captum)
- **Counterfactual Explanations** 
- **Fairness Metrics** (group metrics, statistical parity, equalized odds)
- **Automated Audit Reports** (professional HTML reports)

**Production-Ready**: Built with industry-standard libraries and designed for real-world deployment scenarios.

## 🚀 Features

### 🔍 Explainability
- **SHAP Integration**: TreeExplainer and KernelExplainer support for sklearn models
- **Captum Support**: Integrated Gradients, DeepLift, and GradientShap for PyTorch models
- **Visualizations**: Summary plots, waterfall plots, and feature importance charts
- **Multi-format Support**: Works with both sklearn and PyTorch models

### 🔄 Counterfactual Explanations
- **Nearest Neighbor CFs**: Generate counterfactual explanations using similarity-based approaches
- **Protected Features**: Specify features that should not be modified (e.g., sensitive attributes)
- **Quality Metrics**: Evaluate counterfactuals with validity, proximity, and sparsity metrics
- **Batch Processing**: Generate counterfactuals for multiple instances efficiently

### ⚖️ Fairness Metrics
- **Group Fairness**: Statistical parity difference, equal opportunity, equalized odds
- **Disparate Impact**: Calculate disparate impact ratios with legal thresholds
- **Subgroup Analysis**: Detailed fairness metrics across different population segments
- **Comprehensive Auditing**: Automated fairness assessment with recommendations

### 📊 Audit Reports
- **Professional HTML Reports**: Comprehensive audit-ready documentation
- **Interactive Visualizations**: Charts and plots for better understanding
- **Executive Summary**: High-level fairness assessment and recommendations
- **Customizable**: Tailor reports for different stakeholders

## 🛠️ Tech Stack

- **Core ML**: PyTorch, scikit-learn
- **Explainability**: SHAP, Captum (PyTorch IG)
- **Fairness**: Custom implementation (AIF360 compatible)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Development**: Jupyter notebooks, pytest

## 📁 Repository Structure

```
interpretable-fairness-toolkit/
├─ examples/
│  └─ demo_classification.ipynb    # Complete workflow demo
├─ src/
│  ├─ explainability.py            # SHAP & Captum wrappers
│  ├─ counterfactuals.py           # Nearest neighbor CFs
│  ├─ fairness.py                  # Group metrics & fairness analysis
│  └─ report.py                    # HTML audit report generation
├─ tests/
│  ├─ test_fairness.py            # Fairness module tests
│  ├─ test_explainability.py      # Explainability module tests
│  └─ test_counterfactuals.py     # Counterfactuals module tests
├─ requirements.txt               # Dependencies
└─ README.md                      # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Interpretable-Fairness-Aware-ML-Toolkit.git
cd Interpretable-Fairness-Aware-ML-Toolkit

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import sys
sys.path.append('src')

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import explainability, fairness, counterfactuals, report

# 1. Train a model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 2. Generate explanations
explainer = explainability.Explainer(model, "sklearn", X[:100])
explanations = explainer.explain_instance(X[:5])

# 3. Evaluate fairness
sensitive_attr = np.random.choice([0, 1], size=len(X))
auditor = fairness.FairnessAuditor({'sensitive_attr': sensitive_attr})
fairness_results = auditor.audit(y, model.predict(X))

# 4. Generate audit report
report.generate_audit_report(fairness_results, explanations, output_path="audit_report.html")
```

## 📖 Detailed Examples

### Complete Workflow Demo

Run the comprehensive demo notebook:

```bash
jupyter notebook examples/demo_classification.ipynb
```

This notebook covers:
- Data preparation with sensitive attributes
- Model training and evaluation
- SHAP explanations and visualizations
- Counterfactual generation with protected features
- Comprehensive fairness analysis
- Professional audit report generation

### Explainability Example

```python
# Initialize explainer
explainer = explainability.Explainer(
    model=model, 
    model_type="sklearn",
    background_data=X_train[:100]
)

# Generate SHAP explanations
explanations = explainer.explain_instance(X_test[:5], method="shap")

# Visualize feature importance
explainer.plot_summary(explanations, feature_names=feature_names)
```

### Counterfactual Example

```python
# Generate counterfactuals
cf_generator = counterfactuals.CounterfactualGenerator(
    model=model,
    X_background=X_train,
    y_background=y_train,
    protected_features=['sensitive_attr']  # Don't modify sensitive features
)

cf_result = cf_generator.generate_counterfactual(
    X=X_test[0],
    y_original=y_pred[0],
    max_distance=1.0,
    num_candidates=3
)
```

### Fairness Analysis Example

```python
# Comprehensive fairness audit
auditor = fairness.FairnessAuditor(
    sensitive_attrs={'gender': gender_attr, 'race': race_attr},
    privileged_groups={'gender': 1, 'race': 0}
)

results = auditor.audit(y_true, y_pred, y_prob)

# Get recommendations
recommendations = auditor.generate_recommendations()
```

## 🔧 Configuration

### Model Support

**Supported Models:**
- **sklearn**: RandomForest, GradientBoosting, LogisticRegression, etc.
- **PyTorch**: Any nn.Module model (with Captum installed)

**Optional Dependencies:**
- `captum`: For PyTorch model explanations
- `aif360`: For additional fairness metrics (optional)

### Fairness Configuration

```python
# Configure sensitive attributes
sensitive_attrs = {
    'gender': gender_data,           # Binary: 0/1
    'race': race_data,              # Multi-class: 0,1,2...
    'age_group': age_group_data     # Categorical
}

# Configure privileged groups
privileged_groups = {
    'gender': 1,        # Male as privileged
    'race': 0,          # Group 0 as privileged
    'age_group': 'adult'  # Adults as privileged
}
```

## 📊 Fairness Metrics Explained

### Statistical Parity Difference
- **Definition**: Difference in positive prediction rates between groups
- **Ideal Value**: 0 (perfect parity)
- **Range**: [-1, 1]
- **Interpretation**: Values far from 0 indicate bias

### Equal Opportunity Difference
- **Definition**: Difference in true positive rates between groups
- **Ideal Value**: 0 (equal opportunity)
- **Range**: [-1, 1]
- **Use Case**: Important when false negatives are costly

### Disparate Impact Ratio
- **Definition**: Ratio of positive prediction rates (unprivileged/privileged)
- **Legal Threshold**: 0.8 - 1.25 (80% rule)
- **Ideal Value**: 1.0
- **Use Case**: Legal compliance assessment

### Equalized Odds
- **Definition**: Equal true positive and false positive rates across groups
- **Components**: TPR and FPR for each group
- **Use Case**: Comprehensive fairness assessment

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_fairness.py

# Run with coverage
pytest tests/ --cov=src
```

## 📈 Performance Considerations

### Explainability
- **SHAP Background Data**: Use representative subset (50-100 samples)
- **Tree Models**: TreeExplainer is faster than KernelExplainer
- **Batch Processing**: Process multiple instances together

### Counterfactuals
- **Background Dataset**: Larger datasets provide better counterfactuals
- **Distance Threshold**: Adjust based on feature scales
- **Protected Features**: Reduces search space and improves quality

### Fairness Analysis
- **Sample Size**: Minimum 100 samples per group for reliable metrics
- **Multiple Attributes**: Analyze intersectionality carefully
- **Statistical Significance**: Consider confidence intervals for small groups

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/Interpretable-Fairness-Aware-ML-Toolkit.git
cd Interpretable-Fairness-Aware-ML-Toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SHAP**: For explainability algorithms
- **Captum**: For PyTorch model interpretability
- **AIF360**: For fairness metrics inspiration
- **scikit-learn**: For ML model implementations

## 📞 Support

- **Issues**: Report bugs via [GitHub Issues](https://github.com/your-username/Interpretable-Fairness-Aware-ML-Toolkit/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/your-username/Interpretable-Fairness-Aware-ML-Toolkit/discussions)
- **Documentation**: Check our [Wiki](https://github.com/your-username/Interpretable-Fairness-Aware-ML-Toolkit/wiki) for detailed guides

## 🔮 Roadmap

- [ ] **Enhanced Visualization**: Interactive dashboards for fairness metrics
- [ ] **Model Comparison**: Compare fairness across multiple models
- [ ] **Automated Mitigation**: Built-in bias mitigation techniques
- [ ] **API Integration**: RESTful API for enterprise deployment
- [ ] **Extended Model Support**: XGBoost, LightGBM, TensorFlow
- [ ] **Real-time Monitoring**: Fairness drift detection in production

---

**Built for organizations committed to responsible AI development and deployment.**
