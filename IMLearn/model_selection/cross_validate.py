from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_errors = []
    validation_errors = []
    # split X,y into #cv equal sets, fit each set and calculate the error
    folder = np.mod(np.arange(X.shape[0]), cv)  # take every fifth sample to be in validation set
    for s in range(cv):
        # split to train and validate
        X_train, y_train = X[folder != s], y[folder != s]
        X_validate, y_validate = X[folder == s], y[folder == s]
        # fit and predict
        estimator.fit(X_train, y_train)
        y_pred_train = estimator.predict(X_train)
        y_pred_validation = estimator.predict(X_validate)
        # assume we are calling the mean_square_error function
        train_errors.append(scoring(y_train, y_pred_train))
        validation_errors.append(scoring(y_validate, y_pred_validation))
    return np.mean(train_errors),  np.mean(validation_errors)







