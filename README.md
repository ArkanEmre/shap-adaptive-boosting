# SHAP for Adaptive Boosting

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

TreeSHAP-compatible AdaBoost and RUSBoost classifiers for explainable AI on imbalanced datasets.

## Overview

This package provides custom implementations of AdaBoost and RUSBoost classifiers that are compatible with SHAP's TreeExplainer, enabling fast local explanations for boosting algorithms on imbalanced datasets. The implementation is based on research from my master's thesis "A Novel Approach to Explainability in Tree-Based Ensemble Algorithms for Imbalanced Datasets" (Leipzig University and Alexander Thamm GmbH, 2023).

### Key Features

- **TreeSHAP Compatibility**: Modified boosting algorithms that work with SHAP's TreeExplainer
- **Imbalanced Dataset Support**: RUSBoost implementation for handling class imbalance
- **Fast Explanations**: Polynomial time complexity for SHAP value computation
- **Custom Explainers**: Dedicated explainer classes for both algorithms

## 📁 Project Structure

```
.
├── src/
│   └── shap_adaptive_boosting/
│       ├── classifiers.py    # Custom AdaBoost and RUSBoost implementations
│       └── explainers.py     # TreeSHAP-compatible explainers
└── tests/
    ├── test_adaboost.py
    ├── test_adaboost_explainer.py
    ├── test_rusboost.py
    └── test_rusboost_explainer.py
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```
```bash
pre-commit autoupdate
pre-commit install
```

## Usage

### Basic Example

```python
from shap import Explanation
from shap.datasets import adult
from shap.plots import beeswarm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from shap_adaptive_boosting.classifiers import RUSBoostClassifier
from shap_adaptive_boosting.explainers import RUSBoostExplainer

X_train, X_test, y_train, y_test = train_test_split(
    *adult(), test_size=0.2, random_state=42
)

rbc = RUSBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    learning_rate=0.1,
    n_estimators=50,
    random_state=0,
    sampling_strategy=1 / 3,
    replacement=True,
)
rbc.fit(X_train, y_train)

predictions = rbc.predict_proba(X_test)

rbe = RUSBoostExplainer(rbc, X_train)
shap_values = rbe.shap_values(X_test)

beeswarm(
    shap_values=Explanation(
        values=shap_values[:, :, 0],
        base_values=rbe.expected_value[0],
        data=X_test,
        feature_names=adult()[0].columns,
    )
)
```

### Custom Classifiers

- **`AdaBoostClassifier`**: Modified SAMME algorithm compatible with TreeSHAP
- **`RUSBoostClassifier`**: Combines AdaBoost with random undersampling for imbalanced datasets

### Custom Explainers

- **`AdaBoostExplainer`**: Generates SHAP explanations for Custom AdaBoost
- **`RUSBoostExplainer`**: Generates SHAP explanations for Custom RUSBoost

## Testing

Run the test suite:
```bash
poetry run pytest
```

## Technical Details

### TreeSHAP Integration

The implementation modifies the `predict_proba` method of AdaBoost and RUSBoost to use linear probability calculations instead of log-transformed probabilities, ensuring compatibility with TreeSHAP's framework.

### SHAP Value Computation

Both explainers calculate SHAP values by:
1. Creating individual TreeExplainer instances for each weak learner
2. Computing weighted averages of SHAP values across the ensemble
3. Maintaining compatibility with SHAP's visualization tools
