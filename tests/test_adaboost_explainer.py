import numpy as np
from shap import summary_plot
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from shap_adaptive_boosting.classifiers import AdaBoostClassifier
from shap_adaptive_boosting.explainers import AdaBoostExplainer

RANDOM_STATE = 0
TEST_SIZE = 1 / 3
MULTI_N_CLASSES = 5

# Generate synthetic binary data
X_BINARY, Y_BINARY = make_classification(
    n_samples=2000, random_state=RANDOM_STATE
)
X_BINARY_TRAIN, X_BINARY_TEST, Y_BINARY_TRAIN, Y_BINARY_TEST = train_test_split(
    X_BINARY, Y_BINARY, test_size=TEST_SIZE, stratify=Y_BINARY
)

# Generate synthetic multi-class data
X_MULTI, Y_MULTI = make_classification(
    n_samples=1000 * MULTI_N_CLASSES,
    n_features=MULTI_N_CLASSES * 3,
    n_classes=MULTI_N_CLASSES,
    n_informative=MULTI_N_CLASSES * 2,
    random_state=RANDOM_STATE,
)
X_MULTI_TRAIN, X_MULTI_TEST, Y_MULTI_TRAIN, Y_MULTI_TEST = train_test_split(
    X_MULTI, Y_MULTI, test_size=TEST_SIZE, stratify=Y_MULTI
)


def test_adaboost_explainer_interventional_sum_match() -> None:
    """Test SHAP values sum to model output for interventional perturbation.

    Verifies that SHAP values computed with interventional feature
    perturbation plus the expected value equal the model's predict_proba
    output. Tests this property on both binary and multi-class datasets.
    """

    def _test_sum_match_on_dataset(X_train, X_test, y_train) -> None:
        ab = AdaBoostClassifier(
            random_state=RANDOM_STATE,
        )
        ab.fit(X=X_train, y=y_train)

        ab_preds = ab.predict_proba(X=X_test)

        abe = AdaBoostExplainer(
            model=ab,
            data=X_train,
            model_output="raw",
            feature_perturbation="interventional",
        )
        shap_values = abe.shap_values(X=X_test)
        assert np.allclose(
            shap_values.sum(1) + abe.expected_value, ab_preds, atol=1e-4
        ), "SHAP values don't sum to model output!"

    _test_sum_match_on_dataset(
        X_train=X_BINARY_TRAIN,
        X_test=X_BINARY_TEST,
        y_train=Y_BINARY_TRAIN,
    )
    _test_sum_match_on_dataset(
        X_train=X_MULTI_TRAIN,
        X_test=X_MULTI_TEST,
        y_train=Y_MULTI_TRAIN,
    )


def test_adaboost_explainer_tree_path_dependent_sum_match() -> None:
    """Test SHAP values sum to model output for interventional perturbation.

    Verifies that SHAP values computed with interventional feature
    perturbation plus the expected value equal the model's predict_proba
    output. Tests this property on both binary and multi-class datasets.
    """

    def _test_sum_match_on_dataset(X_train, X_test, y_train, y_test) -> None:
        ab = AdaBoostClassifier(
            random_state=RANDOM_STATE,
        )
        ab.fit(X=X_train, y=y_train)

        ab_preds = ab.predict_proba(X=X_test)

        abe = AdaBoostExplainer(
            model=ab,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )
        shap_values = abe.shap_values(X=X_test)
        assert np.allclose(
            shap_values.sum(1) + abe.expected_value, ab_preds, atol=1e-4
        ), "SHAP values don't sum to model output!"

        shap_interaction_values = abe.shap_interaction_values(
            X=X_test, y=y_test
        )
        assert np.allclose(
            shap_interaction_values.sum(axis=(1, 2)) + abe.expected_value,
            ab_preds,
            atol=1e-4,
        ), "SHAP interaction values don't sum to model output!"

    _test_sum_match_on_dataset(
        X_train=X_BINARY_TRAIN,
        X_test=X_BINARY_TEST,
        y_train=Y_BINARY_TRAIN,
        y_test=Y_BINARY_TEST,
    )
    _test_sum_match_on_dataset(
        X_train=X_MULTI_TRAIN,
        X_test=X_MULTI_TEST,
        y_train=Y_MULTI_TRAIN,
        y_test=Y_MULTI_TEST,
    )


def test_adaboost_explainer_interaction_values_symmetry() -> None:
    """Test the symmetry property of SHAP interaction values.

    Verifies that SHAP interaction values are symmetric, meaning that the
    interaction between feature i and j equals the interaction between
    feature j and i. Also ensures interaction plots can be generated without
    errors. Tests on both binary and multi-class datasets.
    """

    def _test_symmetry_on_dataset(X_train, X_test, y_train, y_test) -> None:
        ab = AdaBoostClassifier(
            random_state=RANDOM_STATE,
        )
        ab.fit(X=X_train, y=y_train)

        abe = AdaBoostExplainer(
            model=ab,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )

        shap_interaction_values = abe.shap_interaction_values(
            X=X_test, y=y_test
        )
        assert np.allclose(
            shap_interaction_values, np.swapaxes(shap_interaction_values, 1, 2)
        )

        # ensure the interaction plot works
        for c in range(ab.n_classes_):
            summary_plot(
                shap_interaction_values[:, :, :, c], X_test, show=False
            )

    _test_symmetry_on_dataset(
        X_train=X_BINARY_TRAIN,
        X_test=X_BINARY_TEST,
        y_train=Y_BINARY_TRAIN,
        y_test=Y_BINARY_TEST,
    )
    _test_symmetry_on_dataset(
        X_train=X_MULTI_TRAIN,
        X_test=X_MULTI_TEST,
        y_train=Y_MULTI_TRAIN,
        y_test=Y_MULTI_TEST,
    )
