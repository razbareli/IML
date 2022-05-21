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
    folder = np.remainder(np.arange(X.shape[0]), cv)
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

    return (np.mean(train_errors),  np.mean(validation_errors))






if __name__ == '__main__':
    X = np.array([[74,99,66],[23,64,1],[56,45,5],[74,99,66],[23,64,1],[56,45,5]])
    # print(X)
    y = np.array([234,23,65,345,456,567])
    # print(y)
    # size_of_set = int(X.shape[0] / 3)
    # splits = np.linspace(size_of_set, X.shape[0] -size_of_set ,size_of_set).astype(int)
    # print(splits)
    #
    # X_S = np.split(X ,splits)
    # y_S = np.split(y ,splits)
    # for i in range(len(X_S)):
    #     print(X_S[i], "->",y_S[i])
    # cv = 3
    # folder = np.remainder(np.arange(X.shape[0]), cv)
    # for s in range(cv):
    #     X_train, y_train = X[folder != s], y[folder != s]
    #     X_validate, y_validate = X[folder == s], y[folder == s]
    #     print("round = ", s)
    #     print(X_train)
