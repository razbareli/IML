from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

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
    samples = np.random.uniform(-1.2, 2, n_samples)
    y = f(samples)
    X = y + epsilon

    # split to train and test
    X_train, y_train, X_test, y_test = split_train_test(X, y, .66)

    # plot
    fig_1 = go.Figure()
    fig_1.add_trace(go.Scatter(x=samples, y=y, mode="markers",
                               marker=dict(color="black"), name="True Model", showlegend=True))
    fig_1.add_trace(go.Scatter(x=samples, y=X, mode="markers",
                               marker=dict(color="red"), name="Noise Model", showlegend=True))
    fig_1.update_layout(title=f"Graph of Polynom f with {n_samples} samples <br> True Model and Noisy model with Noise"
                              f" = {noise}",
                        xaxis_title="samples sampled uniformly from [-1.2, 2]",
                        yaxis_title="f(x) for f = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)",
                        title_x=0.5)
    fig_1.show()

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
    fig_2.add_trace(go.Scatter(x=pol_deg, y=train_errors, mode="markers",
                               marker=dict(color="red"), name="Average Train Error", showlegend=True))
    fig_2.add_trace(go.Scatter(x=pol_deg, y=validation_errors, mode="markers",
                               marker=dict(color="blue"), name="Average Validation Error", showlegend=True))

    fig_2.update_layout(title="Average of Train & Validation Errors in 5-Fold Cross-Validation",
                        xaxis_title="Polynomial Degree",
                        yaxis_title="Average Error",
                        title_x=0.5)
    fig_2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_error = min(validation_errors)
    k_of_min_error = validation_errors.index(min_error)
    best_estimator = PolynomialFitting(k_of_min_error).fit(X_train, y_train)
    test_error = mean_square_error(y_test, best_estimator.predict(X_test))
    print("K* = ", k_of_min_error)
    print("test error for K* prediction = ", test_error)


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
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
