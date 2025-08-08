"""SHAP explainers for AdaBoost and RUSBoost classifiers.

This module provides TreeSHAP-based explainer classes for computing SHAP
values and SHAP interaction values for AdaBoostClassifier and
RUSBoostClassifier models. The explainers support both interventional
and tree path dependent feature perturbation methods.

Typical usage example:

    from shap_adaptive_boosting.explainers import AdaBoostExplainer

    explainer = AdaBoostExplainer(model=trained_model, data=X_train)
    shap_values = explainer.shap_values(X_test)
"""

import warnings
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from shap import TreeExplainer
from shap.utils import safe_isinstance
from shap.utils._exceptions import InvalidModelError
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from shap_adaptive_boosting.classifiers import (
    AdaBoostClassifier,
    RUSBoostClassifier,
)


class AdaBoostExplainer:
    """Explainer class to explain an AdaBoostClassifier with TreeSHAP."""

    def __init__(
        self,
        model: AdaBoostClassifier,
        data: npt.ArrayLike = None,
        model_output: str = "raw",
        feature_perturbation: str = "interventional",
        feature_names: List = None,
    ) -> None:
        """Initialize an instance of AdaBoostExplainer.

        Args:
            model (AdaBoostClassifier):
                The AdaBoostClassifier model to be explained.
            data (npt.ArrayLike, optional):
                The background data used for feature perturbation. Defaults to
                None.
            model_output (str, optional):
                The type of model output to be explained ("raw" or
                "probability"). Defaults to "raw".
            feature_perturbation (str, optional):
                The type of feature perturbation method to be used
                ("interventional" or "tree_path_dependent"). Defaults to
                "interventional".
            feature_names (List, optional):
                The list of feature names. Defaults to None.

        Raises:
            InvalidModelError:
                If the provided model is not an instance of
                models.tree.AdaBoostClassifier.

        """
        if not safe_isinstance(
            obj=model,
            class_path_str=(
                "shap_adaptive_boosting.classifiers.AdaBoostClassifier"
            ),
        ):
            raise InvalidModelError(
                "Only shap_adaptive_boosting.classifiers.AdaBoostClassifier is"
                f" supported. Not {str(type(model))!r}"
            )
        check_is_fitted(estimator=model)
        self.model = model
        self.model_weights = self.model.estimator_weights_[
            : len(self.model.estimators_)
        ]
        self.data = data
        if self.data is None:
            feature_perturbation = "tree_path_dependent"
            warnings.warn(
                message=(
                    'Setting feature_perturbation = "tree_path_dependent"'
                    " because no background data was given."
                )
            )
        elif (
            feature_perturbation == "interventional"
            and self.data.shape[0] > 1000
        ):
            warnings.warn(
                message="Passing "
                + str(self.data.shape[0])
                + " background samples may lead to slow runtimes. Consider"
                " using shap.sample(data, 100) to create a smaller background"
                " data set."
            )
        self.feature_perturbation = feature_perturbation
        self.data_missing = None if self.data is None else pd.isna(self.data)
        self.feature_names = feature_names
        self.model_output = model_output
        self.explainers = []
        self._construct_explainers_and_expected_value()

    def _construct_explainers_and_expected_value(self):
        """Constructs explainers and computes expected value for AdaBoost."""
        self.explainers = []
        expected_values = []
        for estimator in self.model.estimators_:
            explainer = TreeExplainer(
                model=estimator,
                feature_perturbation=self.feature_perturbation,
                data=self.data,
                feature_names=self.feature_names,
                model_output=self.model_output,
            )
            self.explainers.append(explainer)
            expected_values.append(explainer.expected_value)
        self.expected_value = np.average(
            a=expected_values, weights=self.model_weights, axis=0
        )

    def shap_values(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike = None,
        approximate: bool = False,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """Calculate SHAP values for the given input data.

        Args:
            X (npt.ArrayLike):
                The input data for which SHAP values are calculated.
            y (npt.ArrayLike, optional):
                The target values for the input data. Defaults to None.
            approximate (bool, optional):
                Whether to use approximate SHAP values. Defaults to False.
            check_additivity (bool, optional):
                Whether to check the additivity of the SHAP values. Defaults to
                False.

        Returns:
            npt.ArrayLike: The calculated SHAP values.

        """
        shap_values_list = []
        for explainer in tqdm(self.explainers):
            shap_values_list.append(
                explainer.shap_values(
                    X=X,
                    y=y,
                    approximate=approximate,
                    check_additivity=check_additivity,
                )
            )
        return np.average(
            a=shap_values_list, weights=self.model_weights, axis=0
        )

    def shap_interaction_values(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike = None,
    ) -> np.ndarray:
        """Calculate SHAP interaction values for the given input data.

        Args:
            X (npt.ArrayLike):
                The input data for which SHAP interaction values are calculated.
            y (npt.ArrayLike, optional):
                The target values for the input data. Defaults to None.

        Returns:
            npt.ArrayLike: The calculated SHAP interaction values.

        Raises:
            ValueError:
                If the feature_perturbation parameter is set to
                "interventional".

        """
        if self.feature_perturbation == "interventional":
            raise ValueError(
                "The 'feature_perturbation' parameter of"
                f" {self.__class__.__name__!r} was set to"
                f" {self.feature_perturbation!r}.\nInteraction values are"
                " currently not supported for'feature_perturbation' =="
                " 'interventional'."
            )
        shap_interaction_values = []
        for explainer in tqdm(self.explainers):
            shap_interaction_values.append(
                explainer.shap_interaction_values(X=X, y=y)
            )
        return np.average(
            a=shap_interaction_values, weights=self.model_weights, axis=0
        )


class RUSBoostExplainer(AdaBoostExplainer):
    """Explainer class to explain a RUSBoostClassifier with TreeSHAP."""

    def __init__(
        self,
        model: RUSBoostClassifier,
        data: npt.ArrayLike = None,
        model_output: str = "raw",
        feature_perturbation: str = "interventional",
        feature_names: List = None,
        rus_background_data: bool = False,
        background_data_Y: npt.ArrayLike = None,
    ) -> None:
        """Initialize an instance of RUSBoostExplainer.

        Args:
            model (RUSBoostClassifier):
                The RUSBoostClassifier model to be explained.
            data (npt.ArrayLike, optional):
                The background data used for feature perturbation. Defaults to
                None.
            model_output (str, optional):
                The type of model output to be explained ("raw" or
                "probability"). Defaults to "raw".
            feature_perturbation (str, optional):
                The type of feature perturbation method to be used
                ("interventional" or "tree_path_dependent"). Defaults to
                "interventional".
            feature_names (List, optional):
                The list of feature names. Defaults to None.
            rus_background_data (bool, optional):
                Whether to use random-under-sampled subsets of each tree as
                background data. Defaults to False.
            background_data_Y (npt.ArrayLike, optional):
                The target variable for the background data. Required if
                rus_background_data is True. Defaults to None.

        Raises:
            InvalidModelError:
                If the provided model is not an instance of
                shap_adaptive_boosting.classifiers.RUSBoostClassifier.
            ValueError:
                If rus_background_data is True but data or background_data_Y is
                not provided.

        """
        if not safe_isinstance(
            obj=model,
            class_path_str=(
                "shap_adaptive_boosting.classifiers.RUSBoostClassifier"
            ),
        ):
            raise InvalidModelError(
                "Only shap_adaptive_boosting.classifiers.RUSBoostClassifier is"
                f" supported. Not {str(type(model))!r}"
            )
        if rus_background_data and (background_data_Y is None or data is None):
            raise ValueError(
                "To use the random-under-sampled subsets of each tree as"
                " background data, the entire training dataset and target"
                " variable have to be provided via 'data' and"
                " 'background_data_y', respectively."
            )
        self.rus_background_data = rus_background_data
        self.background_data_y = background_data_Y
        super().__init__(
            model=model,
            data=data,
            model_output=model_output,
            feature_perturbation=feature_perturbation,
            feature_names=feature_names,
        )

    def _construct_explainers_and_expected_value(self):
        """Constructs explainers and computes expected value for RUSBoost."""
        self.explainers = []
        expected_values = []
        for estimator, sampler in zip(
            self.model.estimators_, self.model.samplers_
        ):
            explainer = TreeExplainer(
                model=estimator,
                feature_perturbation=self.feature_perturbation,
                data=sampler.fit_resample(self.data, self.background_data_y)[0]
                if self.rus_background_data
                else self.data,
                feature_names=self.feature_names,
                model_output=self.model_output,
            )
            self.explainers.append(explainer)
            expected_values.append(explainer.expected_value)

        self.expected_value = np.average(
            a=expected_values, weights=self.model_weights, axis=0
        )
