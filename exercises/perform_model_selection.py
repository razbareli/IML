from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # generate dataset
    mu = 0
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    epsilon = np.random.normal(mu, noise, n_samples)
    X = np.linspace(-1.2, 2, n_samples)
    y_no_noise = f(X)
    y = y_no_noise + epsilon

    # split to train and test
    X_train, y_train, X_test, y_test = split_train_test(X, y, 2 / 3)

    # plot
    fig_1 = go.Figure()
    fig_1.add_trace(go.Scatter(x=X, y=y_no_noise, mode="markers",
                               marker=dict(color="black"), name="True Model", showlegend=True))
    fig_1.add_trace(go.Scatter(x=X, y=y, mode="markers",
                               marker=dict(color="red"), name="Noise Model", showlegend=True))
    fig_1.update_layout(title=f"Graph of Polynom f with {n_samples} samples <br> True Model and Noisy model with Noise"
                              f" = {noise}",
                        xaxis_title="samples sampled uniformly from [-1.2, 2]",
                        yaxis_title="f(x) for f = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)",
                        title_x=0.5)
    # fig_1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    pol_deg = [i for i in range(11)]
    train_errors = []
    validation_errors = []
    for k in pol_deg:
        estimator = PolynomialFitting(k)
        t_err, v_err = cross_validate(estimator, X_train, y_train, mean_square_error, 5)
        train_errors.append(t_err)
        validation_errors.append(v_err)
    fig_2 = go.Figure()
    fig_2.add_trace(go.Scatter(x=pol_deg, y=train_errors, mode="markers+lines",
                               marker=dict(color="red"), name="Average Train Error", showlegend=True))
    fig_2.add_trace(go.Scatter(x=pol_deg, y=validation_errors, mode="markers+lines",
                               marker=dict(color="blue"), name="Average Validation Error", showlegend=True))

    fig_2.update_layout(title="Average of Train & Validation Errors in 5-Fold Cross-Validation",
                        xaxis_title="Polynomial Degree",
                        yaxis_title="Average Error",
                        title_x=0.5)
    # fig_2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_error = min(validation_errors)
    k_of_min_error = validation_errors.index(min_error)
    best_estimator = PolynomialFitting(k_of_min_error).fit(X_train, y_train)
    test_error = mean_square_error(y_test, best_estimator.predict(X_test))
    print(f"noise: {noise}, n_samples: {n_samples}")
    print(f"K* = {k_of_min_error}")
    print(f"Test Error = {test_error}")
    print("-------------------------------------------------------------")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_size = 50
    X_train = X[:train_size, :]
    y_train = y[:train_size]
    X_test = X[train_size:, :]
    y_test = y[train_size:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ranges = np.linspace(0.001, 2, num=n_evaluations)
    train_errors_ridge = []
    val_errors_ridge = []
    train_errors_lasso = []
    val_errors_lasso = []
    for lam in ranges:
        ridge = RidgeRegression(lam)
        lasso = Lasso(alpha=lam)
        t_err_ridge, v_err_ridge = cross_validate(ridge, X_train, y_train, mean_square_error, 5)
        t_err_lasso, v_err_lasso = cross_validate(lasso, X_train, y_train, mean_square_error, 5)
        train_errors_ridge.append(t_err_ridge)
        val_errors_ridge.append(v_err_ridge)
        train_errors_lasso.append(t_err_lasso)
        val_errors_lasso.append(v_err_lasso)

    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=ranges, y=train_errors_ridge, mode="lines",
                               marker=dict(color="red"), name="Train Ridge", showlegend=True))
    fig_3.add_trace(go.Scatter(x=ranges, y=val_errors_ridge, mode="lines",
                               marker=dict(color="blue"), name="Validation Ridge", showlegend=True))
    fig_3.add_trace(go.Scatter(x=ranges, y=train_errors_lasso, mode="lines",
                               marker=dict(color="green"), name="Train Lasso", showlegend=True))
    fig_3.add_trace(go.Scatter(x=ranges, y=val_errors_lasso, mode="lines",
                               marker=dict(color="black"), name="Validation Lasso", showlegend=True))
    fig_3.update_layout(title=f"Error Rates for Different Regularization Parameters <br> "
                              f"n_samples = {n_samples}, n_evaluations = {n_evaluations},",
                        xaxis_title="Lambda",
                        yaxis_title="Average Error",
                        title_x=0.5)
    fig_3.show()
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_error_ridge = min(val_errors_ridge)
    min_error_lasso = min(val_errors_lasso)
    best_lam_ridge = ranges[val_errors_ridge.index(min_error_ridge)]
    best_lam_lasso = ranges[val_errors_lasso.index(min_error_lasso)]

    best_ridge = Ridge(best_lam_ridge)
    best_ridge.fit(X_train, y_train)
    best_lasso = Lasso(best_lam_lasso)
    best_lasso.fit(X_train, y_train)
    least_squares = LinearRegression().fit(X_train, y_train)

    test_error_ridge = mean_square_error(y_test, best_ridge.predict(X_test))
    test_error_lasso = mean_square_error(y_test, best_lasso.predict(X_test))
    test_error_least_squares = mean_square_error(y_test, least_squares.predict(X_test))

    print(f"Ridge: best lambda = {best_lam_ridge}, test error = {test_error_ridge}")
    print(f"Lasso: best lambda = {best_lam_lasso}, test error = {test_error_lasso}")
    print(f"Least Squares: test error = {test_error_least_squares}")
    print("-------------------------------------------------------------")


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(1500, 10)
    select_regularization_parameter()
