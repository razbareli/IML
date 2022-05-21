import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    test_errors = []
    train_errors = []
    for T in range(n_learners):
        train_errors.append(adaboost.partial_loss(train_X, train_y, T))
        test_errors.append(adaboost.partial_loss(test_X, test_y, T))

    # plot
    fig_1 = go.Figure()
    fig_1.add_trace(go.Scatter(x=[i for i in range(n_learners)], y=test_errors, mode="lines",
                             marker=dict(color="blue"), name="Test Error", showlegend=True))
    fig_1.add_trace(go.Scatter(x=[i for i in range(n_learners)], y=train_errors, mode="lines",
                             marker=dict(color="red"), name="Train Error", showlegend=True))

    fig_1.update_layout(title="Error of Adaboost as Function of Number of Weak Learners",
                      xaxis_title="Num Of Learners", yaxis_title="Error")
    fig_1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig_2 = make_subplots(2, 2, subplot_titles=[f"Num of Iterations =  {t}" for t in T])
    for i, t in enumerate(T):
        partial_pred = lambda x: adaboost.partial_predict(x, t)
        fig_2.add_traces([decision_surface(partial_pred, lims[0], lims[1], showscale=False),
                           go.Scatter(x=test_X[:, 0], y=test_X[:, 1], marker=dict(color=test_y),
                                      mode="markers", showlegend=False)], rows=int(i // 2) + 1, cols=(i % 2) + 1)
    fig_2.update_layout(title="Adaboost Prediction for Different Numbers of Iterations", title_x=0.5)
    fig_2.show()

    # Question 3: Decision surface of best performing ensemble
    min_error = min(test_errors)
    T_of_min_error = test_errors.index(min_error)
    from IMLearn.metrics import accuracy
    acc = accuracy(test_y, adaboost.partial_predict(test_X, T_of_min_error))
    # print(T_of_min_error, acc)
    fig_3 = go.Figure()
    fig_3.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, T_of_min_error),
                                        lims[0], lims[1], showscale=False),
                       go.Scatter(x=test_X[:, 0], y=test_X[:, 1], marker=dict(color=test_y),
                                  mode="markers", showlegend=False)])
    fig_3.update_layout(title=f"Adaboost Prediction for {T_of_min_error} Iterations <br> Accuracy = {acc}",
                         title_x=0.5)
    fig_3.show()

    # Question 4: Decision surface with weighted samples
    D_T = adaboost.D_
    D_T = (D_T / np.max(D_T)) * 10  # multiply by 5 as requested was too small
    fig_4 = go.Figure()
    fig_4.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, n_learners),
                                       lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], marker=dict(color=train_y, size=D_T),
                                 mode="markers", showlegend=False)])
    fig_4.update_layout(title=f"Adaboost Prediction for {n_learners} Iterations",
                        title_x=0.5)
    fig_4.show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)


