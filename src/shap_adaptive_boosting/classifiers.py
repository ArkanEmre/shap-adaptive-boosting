"""AdaBoost and RUSBoost classifiers compatible with TreeSHAP.

This module provides custom implementations of AdaBoostClassifier and
RUSBoostClassifier that are compatible with SHAP's TreeExplainer. These
implementations modify the prediction methods to use linear probability
instead of log-transformed probabilities, ensuring proper SHAP value
computation.

Typical usage example:

    from adaptive_boosting_shap.classifiers import AdaBoostClassifier

    classifier = AdaBoostClassifier(n_estimators=50)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict_proba(X_test)
"""

from typing import Any, Callable, Dict, Generator, Union

import numpy as np
import numpy.typing as npt
from imblearn.ensemble import RUSBoostClassifier as RUSBoostClassifierBase
from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifierBase
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class MethodHiderMixin(object):
    """Mixin to hide specific methods from a class interface.

    This mixin intercepts attribute access and raises AttributeError for
    specified method names, effectively hiding them from the public API.
    Used to prevent access to decision_function and staged_decision_function
    methods in AdaBoost classifiers for TreeSHAP compatibility.
    """

    def __getattribute__(self, name) -> Any:
        """Intercept attribute access and hide specific methods.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The requested attribute if it's not in the hidden list.

        Raises:
            AttributeError: If the requested attribute is decision_function
                or staged_decision_function.
        """
        if name in ["decision_function", "staged_decision_function"]:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}"
            )
        return super().__getattribute__(name)


class AdaBoostClassifier(AdaBoostClassifierBase, MethodHiderMixin):
    """AdaBoostClassifier for compatibility with TreeSHAP."""

    def __init__(
        self,
        estimator: BaseDecisionTree = DecisionTreeClassifier(max_depth=1),
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> None:
        """Initialize an instance of AdaBoostClassifier.

        Args:
            estimator (BaseDecisionTree, optional):
                The base decision tree estimator. Defaults to
                DecisionTreeClassifier().
            n_estimators (int, optional):
                The number of estimators in the ensemble. Default is 50.
            learning_rate (float, optional):
                The learning rate for the boosting algorithm. Default is 1.0.
            random_state (Union[int, np.random.RandomState, None], optional):
                Controls the random seed given at each `estimator` at each
                boosting iteration. Default is None.

        Raises:
            InvalidModelError:
                If the provided model is not an instance of
                models.tree.CustomAdaBoostClassifier.
            ValueError: If the algorithm parameter of the model is not "SAMME".

        """
        if not isinstance(estimator, DecisionTreeClassifier):
            raise ValueError(
                f"The 'estimator' parameter of {self.__class__.__name__!r} must"
                f" be 'DecisionTreeClassifier'. Got {estimator!r} instead."
            )
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def _validate_estimator(self) -> None:
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator()
        if not hasattr(self.estimator_, "predict_proba"):
            raise TypeError(
                "AdaBoostClassifier requires "
                "that the weak learner supports the calculation of class "
                "probabilities with a predict_proba method.\n"
                "Please change the estimator."
            )

    def predict_proba(
        self, X: Union[npt.ArrayLike, sparse.spmatrix]
    ) -> np.ndarray:
        """Predict class probabilities for X.

        Compute the linear probability of `X`. The predicted class
        probabilities of an input sample is computed as the weighted mean
        predicted class probabilities of the classifiers in the ensemble. Here
        the probabilities do not undergo a log transformation and softmax.

        Args:
            X (Union[npt.ArrayLike, sparse.spmatrix]): The input samples.

        Returns:
            p (np.ndarray):
                The class probabilities of the input samples. The order of
                outputs is the same as that of the `classes_` attribute.
        """
        check_is_fitted(estimator=self)
        X = self._check_X(X)
        n_classes = self.n_classes_

        if n_classes == 1:
            return np.ones(shape=(super()._num_samples(X), 1))

        proba = np.array(
            object=[
                estimator.predict_proba(X) for estimator in self.estimators_
            ]
        )

        return np.average(
            a=proba,
            weights=self.estimator_weights_[: len(self.estimators_)],
            axis=0,
        )

    def predict(self, X: Union[npt.ArrayLike, sparse.spmatrix]) -> np.ndarray:
        """Predict classes for X.

        Predict classes for `X` using the linear probability. The predicted
        class of an input sample is computed as the weighted mean prediction of
        the classifiers in the ensemble. Here the probabilities do not undergo a
        log transformation and softmax.

        Args:
            X (Union[npt.ArrayLike, sparse.spmatrix]): The input samples.

        Returns:
            y (np.ndarray): The predicted classes.
        """
        return self.predict_proba(X=X).argmax(axis=1)

    def staged_predict_proba(
        self, X: Union[npt.ArrayLike, sparse.spmatrix]
    ) -> Generator[np.ndarray, None, None]:
        """Predict class probabilities for X.

        Compute decision function of `X` for each boosting iteration. This
        method allows monitoring (i.e., determining error on the testing set)
        after each boosting iteration.

        Args:
            X (Union[npt.ArrayLike, sparse.spmatrix]): The input samples.

        Yields:
            score (Generator[np.ndarray]):
                The predicted class probabilities of the input samples. The
                order of outputs is the same as that of the `classes_`
                attribute. Binary classification is a special case with `k ==
                1`, otherwise `k == n_classes`.
        """
        check_is_fitted(estimator=self)
        X = self._check_X(X)

        pred = None
        norm = 0.0

        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            norm += weight

            current_pred = estimator.predict_proba(X)
            current_pred *= weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            yield pred / norm

    def staged_predict(
        self, X: Union[npt.ArrayLike, sparse.spmatrix]
    ) -> Generator[np.ndarray, None, None]:
        """Generate predictions for `X` at each boosting iteration.

        Args:
            X (Union[npt.ArrayLike, sparse.spmatrix]): The input samples.

        Yields:
            predictions (Generator[np.ndarray]):
                The predicted classes at each boosting iteration.
        """
        for predictions in self.staged_predict_proba(X=X):
            yield predictions.argmax(axis=1)


class RUSBoostClassifier(RUSBoostClassifierBase, AdaBoostClassifier):
    """RUSBoostClassifier for compatibility with TreeSHAP."""

    def __init__(
        self,
        estimator: BaseDecisionTree = DecisionTreeClassifier(max_depth=1),
        *,
        n_estimators: int = 50,
        learning_rate: str = 1.0,
        random_state: int = None,
        sampling_strategy: Union[float, str, Dict, Callable] = "auto",
        replacement: bool = False,
    ) -> None:
        """Initialize an instance of RUSBoostClassifier.

        Args:
            estimator (BaseDecisionTree, optional):
                The base decision tree estimator. Default is
                DecisionTreeClassifier().
            n_estimators (int, optional):
                The number of estimators in the ensemble. Default is 50.
            learning_rate (str, optional):
                The learning rate for the boosting algorithm. Default is 1.0.
            random_state (int, optional):
                The random seed for reproducible results. Default is None.
            sampling_strategy (float, str, dict, callable, optional):
                When float (only available for binary classification), it
                corresponds to the desired ratio of the number of samples in the
                minority class over the number of samples in the majority class
                after resampling. When str, specify the class targeted by the
                resampling. The number of samples in the different classes will
                be equalized. Possible choices are: ('majority', 'not minority',
                'not majority', 'all','auto'). When dict, the keys correspond to
                the targeted classes. The values correspond to the desired
                number of samples for each targeted class. When callable,
                function taking y and returns a dict. The keys correspond to the
                targeted classes. The values correspond to the desired number of
                samples for each class. Default is "auto".
            replacement (bool, optional):
                Whether to use replacement during resampling. Default is False.
        """
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
            replacement=replacement,
        )
