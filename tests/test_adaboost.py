from typing import Generator

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from shap_adaptive_boosting.classifiers import AdaBoostClassifier

RANDOM_STATE = 0

# Generate synthetic data
X, y = make_classification(
    n_samples=1000, weights=[1 - 0.05, 0.05], random_state=RANDOM_STATE
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, stratify=y
)


def test_adaboost_estimator() -> None:
    """Test that AdaBoostClassifier's estimator is a DecisionTreeClassifier.

    Verifies that the default estimator attribute of AdaBoostClassifier is
    correctly set to an instance of DecisionTreeClassifier.
    """
    ab = AdaBoostClassifier(random_state=RANDOM_STATE)
    assert isinstance(ab.estimator, DecisionTreeClassifier)


def test_adaboost_predict_proba() -> None:
    """Test that predict_proba equals weighted average of tree predictions.

    Verifies that AdaBoostClassifier's predict_proba method returns values
    that exactly match the weighted average of the individual decision tree
    predictions in the ensemble, using the estimator weights.
    """
    ab = AdaBoostClassifier(
        random_state=RANDOM_STATE,
        estimator=DecisionTreeClassifier(max_depth=10),
    )
    ab.fit(X=X_train, y=y_train)

    ab_preds = ab.predict_proba(X=X_test)
    tree_preds = np.array(
        object=[estimator.predict_proba(X_test) for estimator in ab.estimators_]
    )

    weighted_average_preds = np.average(
        a=tree_preds,
        weights=ab.estimator_weights_[: len(ab.estimators_)],
        axis=0,
    )
    assert np.array_equal(a1=ab_preds, a2=weighted_average_preds), (
        "The output of the AdaBoostClassifier does not equal the weighted"
        " average of the trees."
    )


def test_adaboost_staged_predict_proba() -> None:
    """Test staged_predict_proba returns generator with correct iterations.

    Verifies that the staged_predict_proba method returns a Generator object
    and that it yields exactly one prediction per estimator in the ensemble,
    allowing monitoring of predictions at each boosting iteration.
    """
    ab = AdaBoostClassifier(
        random_state=RANDOM_STATE,
        estimator=DecisionTreeClassifier(max_depth=10),
    )
    ab.fit(X=X_train, y=y_train)

    ab_staged_preds = ab.staged_predict_proba(X=X_test)

    assert isinstance(ab_staged_preds, Generator)
    assert len(list(ab_staged_preds)) == len(ab.estimators_)
