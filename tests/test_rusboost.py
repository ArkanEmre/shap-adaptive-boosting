from typing import Generator

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from shap_adaptive_boosting.classifiers import RUSBoostClassifier

RANDOM_STATE = 0

# Generate synthetic data
X, y = make_classification(
    n_samples=1000, weights=[1 - 0.05, 0.05], random_state=RANDOM_STATE
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, stratify=y
)


def test_rusboost_estimator() -> None:
    """Test that RUSBoostClassifier's estimator is a DecisionTreeClassifier.

    Verifies that the default estimator attribute of RUSBoostClassifier is
    correctly set to an instance of DecisionTreeClassifier.
    """
    rb = RUSBoostClassifier(random_state=RANDOM_STATE)
    assert isinstance(rb.estimator, DecisionTreeClassifier)


def test_rusboost_predict_proba() -> None:
    """Test that predict_proba equals weighted average of tree predictions.

    Verifies that RUSBoostClassifier's predict_proba method returns values
    that exactly match the weighted average of the individual decision tree
    predictions in the ensemble, using the estimator weights.
    """
    rb = RUSBoostClassifier(
        random_state=RANDOM_STATE,
        estimator=DecisionTreeClassifier(max_depth=10),
    )
    rb.fit(X=X_train, y=y_train)

    rb_preds = rb.predict_proba(X=X_test)
    tree_preds = np.array(
        object=[estimator.predict_proba(X_test) for estimator in rb.estimators_]
    )

    weighted_average_preds = np.average(
        a=tree_preds,
        weights=rb.estimator_weights_[: len(rb.estimators_)],
        axis=0,
    )
    assert np.array_equal(a1=rb_preds, a2=weighted_average_preds), (
        "The output of the RUSBoostClassifier does not equal the weighted"
        " average of the trees."
    )


def test_rusboost_staged_predict_proba() -> None:
    """Test staged_predict_proba returns generator with correct iterations.

    Verifies that the staged_predict_proba method returns a Generator object
    and that it yields exactly one prediction per estimator in the ensemble,
    allowing monitoring of predictions at each boosting iteration.
    """
    rb = RUSBoostClassifier(
        random_state=RANDOM_STATE,
        estimator=DecisionTreeClassifier(max_depth=10),
    )
    rb.fit(X=X_train, y=y_train)

    rb_staged_preds = rb.staged_predict_proba(X=X_test)

    assert isinstance(rb_staged_preds, Generator)
    assert len(list(rb_staged_preds)) == len(rb.estimators_)
