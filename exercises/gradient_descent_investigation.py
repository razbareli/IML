import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(model, **kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for func in [L1, L2]:
        for eta in etas:
            initial_weights = np.copy(init)
            norm = func(initial_weights)
            callback, values, weights = get_gd_state_recorder_callback()
            lr = FixedLR(eta)
            GD = GradientDescent(learning_rate=lr, callback=callback)
            best_weights = GD.fit(norm, None, None)
            # plot path
            fig1 = plot_descent_path(module=func, descent_path=np.array(weights), title=f"for eta = {eta}")
            # fig1.show()
            # plot convergence rate
            iterations = [i for i in range(len(values))]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=iterations, y=values, mode="markers",
                                      marker=dict(color="blue"), showlegend=False))
            fig2.update_layout(
                title=f"Convergence Rate for eta = {eta}",
                xaxis_title="iteration",
                yaxis_title="norm value",
                title_x=0.5)
            # fig2.show()

            # loss achived
            # norm.weights = best_weights
            # print(norm.compute_output())


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    # Plot algorithm's convergence for the different values of gamma
    fig_1 = go.Figure()
    colors = ["black", "blue", "green", "red"]
    weights_gamma_95_l1 = None  # to plot the path of gamma = 0.95
    weights_gamma_95_l2 = None  # to plot the path of gamma = 0.95
    for func in [L1, L2]:
        for g in range(len(gammas)):
            initial_weights = np.copy(init)
            norm = func(initial_weights)
            callback, values, weights = get_gd_state_recorder_callback()
            lr = ExponentialLR(eta, gammas[g])
            GD = GradientDescent(learning_rate=lr, callback=callback)
            best_weights = GD.fit(norm, None, None)
            iterations = [i for i in range(len(values))]
            if g == 1 and func == L1:  # to plot the path
                weights_gamma_95_l1 = weights
            if g == 1 and func == L2:  # to plot the path
                weights_gamma_95_l2 = weights
            fig_1.add_trace(go.Scatter(x=iterations, y=values, mode="lines",
                                       marker=dict(color=colors[g]), name="gammas = " + str(gammas[g]),
                                       showlegend=True))
            # loss achived
            # norm.weights = best_weights
            # print(norm.compute_output())

    fig_1.update_layout(
        title=f"Convergence Rate of norm L1 for eta = {eta}",
        xaxis_title="iteration",
        yaxis_title="norm value",
        title_x=0.5)
    # fig_1.show()

    # Plot descent path for gamma=0.95
    fig_2 = plot_descent_path(module=L1, descent_path=np.array(weights_gamma_95_l1), title=f"for eta = {eta}")
    # fig_2.show()
    fig_3 = plot_descent_path(module=L2, descent_path=np.array(weights_gamma_95_l2), title=f"for eta = {eta}")
    # fig_3.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # fitting a logistic regression
    logistic = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    logistic.fit(X_train.to_numpy(), y_train.to_numpy())
    proba = logistic.predict_proba(X_train.to_numpy())

    # roc curve of different alphas
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_train, proba)
    # thresholds = np.linspace(0, 1, 101)
    # fpr, tpr, diff = [], [], []
    # for alpha in thresholds:
    #     y_pred = np.where(proba >= alpha, 1, 0)
    #     fp = np.sum((y_pred == 1) & (y_train == 0))
    #     tp = np.sum((y_pred == 1) & (y_train == 1))
    #     fn = np.sum((y_pred == 0) & (y_train == 1))
    #     tn = np.sum((y_pred == 0) & (y_train == 0))
    #     fpr.append((fp / (fp + tn)))
    #     tpr.append(tp / (tp + fn))
    #     diff.append((tp-fp)/(tp+fn))
    # best_alpha = int(np.argmax(diff)) / 100
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color="red",
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()

    # best alpha
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    print(f"best alpha = {best_alpha}")

    # calculate test error
    prediction_with_best_alpha = proba >= best_alpha
    test_error = misclassification_error(y_train.to_numpy(), prediction_with_best_alpha)
    print(f"Test error with best alpha = {test_error}")

    # Q10 + Q11
    # perform l1, l2 model selection

    train_errors_l1 = []
    val_errors_l1 = []
    train_errors_l2 = []
    val_errors_l2 = []
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for lam in lambdas:
        l1_logistic = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                         penalty="l1", alpha=0.5, lam=lam)
        l2_logistic = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                         penalty="l2", alpha=0.5, lam=lam)
        t_err_l1, v_err_l1 = cross_validate(l1_logistic, X_train.to_numpy(),
                                            y_train.to_numpy(), misclassification_error, 5)
        t_err_l2, v_err_l2 = cross_validate(l2_logistic, X_train.to_numpy(),
                                            y_train.to_numpy(), misclassification_error, 5)
        train_errors_l1.append(t_err_l1)
        val_errors_l1.append(v_err_l1)
        train_errors_l2.append(t_err_l2)
        val_errors_l2.append(v_err_l2)

    min_error_l1 = min(val_errors_l1)
    min_error_l2 = min(val_errors_l2)
    best_lam_l1 = lambdas[val_errors_l1.index(min_error_l1)]
    best_lam_l2 = lambdas[val_errors_l2.index(min_error_l2)]

    best_l1 = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                 penalty="l1", alpha=0.5, lam=best_lam_l1)
    best_l1.fit(X_train.to_numpy(), y_train.to_numpy())
    best_l2 = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                 penalty="l2", alpha=0.5, lam=best_lam_l2)
    best_l2.fit(X_train.to_numpy(), y_train.to_numpy())

    test_error_l1 = round(misclassification_error(y_test.to_numpy(), best_l1.predict(X_test.to_numpy())), 2)
    test_error_l2 = round(misclassification_error(y_test.to_numpy(), best_l2.predict(X_test.to_numpy())), 2)

    print(f"l1: best lambda = {best_lam_l1}, test error = {test_error_l1}")
    print(f"l2: best lambda = {best_lam_l2}, test error = {test_error_l2}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()

