import numpy as np
import pandas as pd
from typing import Tuple, List
from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import split_train_test
from utils import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "simple_white"


def ex7_callback():
    loss = []
    iters = []
    grads = []
    weights = []

    def callback(**kwargs):
        loss.append(kwargs["val"])
        iters.append(kwargs["t"])
        grads.append(np.linalg.norm(kwargs["grad"]))
        if len(iters) > 0 and len(iters) % 100 == 0:
            weights.append(kwargs["weights"])
    return callback, loss, iters, grads, weights

def generate_nonlinear_data(
        samples_per_class: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        train_proportion: float = 0.8) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a multiclass non linearly-separable dataset. Adopted from Stanford CS231 course code.

    Parameters:
    -----------
    samples_per_class: int, default = 100
        Number of samples per class

    n_features: int, default = 2
        Data dimensionality

    n_classes: int, default = 2
        Number of classes to generate

    train_proportion: float, default=0.8
        Proportion of samples to be used for train set

    Returns:
    --------
    train_X : ndarray of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : ndarray of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : ndarray of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : ndarray of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    X, y = np.zeros((samples_per_class * n_classes, n_features)), np.zeros(samples_per_class * n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * 0.2  # theta
        X[ix], y[ix] = np.c_[r * np.sin(t), r * np.cos(t)], j

    split = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion)
    return tuple(map(lambda x: x.values, split))


def plot_decision_boundary(nn: NeuralNetwork, lims, X: np.ndarray = None, y: np.ndarray = None, title=""):
    data = [decision_surface(nn.predict, lims[0], lims[1], density=40, showscale=False)]
    if X is not None:
        col = y if y is not None else "black"
        data += [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                            marker=dict(color=col, colorscale=custom, line=dict(color="black", width=1)))]

    return go.Figure(data,
                     go.Layout(title=rf"$\text{{Network Decision Boundaries {title}}}$",
                               xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400))


def animate_decision_boundary(nn: NeuralNetwork, weights: List[np.ndarray], lims, X: np.ndarray, y: np.ndarray,
                              title="", save_name=None):
    frames = []
    for i, w in enumerate(weights):
        nn.weights = w
        frames.append(go.Frame(data=[decision_surface(nn.predict, lims[0], lims[1], density=40, showscale=False),
                                     go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                                marker=dict(color=y, colorscale=custom,
                                                            line=dict(color="black", width=1)))
                                     ],
                               layout=go.Layout(title=rf"$\text{{{title} Iteration {i + 1}}}$")))

    fig = go.Figure(data=frames[0]["data"], frames=frames[1:],
                    layout=go.Layout(title=frames[0]["layout"]["title"]))
    if save_name:
        animation_to_gif(fig, save_name, 200, width=400, height=400)


if __name__ == '__main__':
    np.random.seed(0)
    # Generate and visualize dataset
    n_features, n_classes = 2, 3
    train_X, train_y, test_X, test_y = generate_nonlinear_data(
        samples_per_class=500, n_features=n_features, n_classes=n_classes, train_proportion=0.8)
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    go.Figure(data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                               marker=dict(color=train_y, colorscale=custom, line=dict(color="black", width=1)))],
              layout=go.Layout(title=r"$\text{Train Data}$", xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400)) \
        .write_image(f"..nonlinear_data.png")

    # ---------------------------------------------------------------------------------------------#
    # Question 1: Fitting simple network with two hidden layers                                    #
    # ---------------------------------------------------------------------------------------------#


    callback, loss_1, iterations_1, grads_1, weights_1 = ex7_callback()
    input_layer = FullyConnectedLayer(n_features, 16, ReLU())
    hidden_layer_1 = FullyConnectedLayer(16, 16, ReLU())
    hidden_layer_2 = FullyConnectedLayer(16, n_classes)
    layers = [input_layer, hidden_layer_1, hidden_layer_2]
    nn_1 = NeuralNetwork(layers, CrossEntropyLoss(),
                       GradientDescent(learning_rate=FixedLR(0.1),
                                       max_iter=5000,
                                       callback=callback))
    nn_1.fit(train_X, train_y)
    pred = nn_1.predict(test_X)
    fig_1 = plot_decision_boundary(nn_1, lims, test_X, test_y)
    fig_1.show()
    print(f"Question 1: accuracy = {accuracy(test_y, pred)}")

    # ---------------------------------------------------------------------------------------------#
    # Question 2: Fitting a network with no hidden layers                                          #
    # ---------------------------------------------------------------------------------------------#
    input_layer = FullyConnectedLayer(n_features, n_classes)
    layers = [input_layer]
    nn_2 = NeuralNetwork(layers, CrossEntropyLoss(),
                       GradientDescent(learning_rate=FixedLR(0.1),
                                       max_iter=5000))
    nn_2.fit(train_X, train_y)
    pred = nn_2.predict(test_X)
    fig_2 = plot_decision_boundary(nn_2, lims, test_X, test_y)
    fig_2.show()
    print(f"Question 2: accuracy = {accuracy(test_y, pred)}")

    # ---------------------------------------------------------------------------------------------#
    # Question 3+4: Plotting network convergence process                                           #
    # ---------------------------------------------------------------------------------------------#

    # Question 3

    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=iterations_1, y=loss_1, mode="lines",
                               marker=dict(color="blue"), name="loss",
                               showlegend=True))
    fig_3.update_layout(
        title=f"Loss as Function of Iteration",
        xaxis_title="iteration",
        title_x=0.5)
    fig_3.show()

    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=iterations_1, y=grads_1, mode="lines",
                               marker=dict(color="red"), name="Gradient Norm",
                               showlegend=True))
    fig_3.update_layout(
        title=f"Gradient Norm as Function of Iteration",
        xaxis_title="iteration",
        title_x=0.5)
    fig_3.show()

    animate_decision_boundary(nn_1, weights_1, lims, test_X, test_y)

    # Question 4
    callback, loss_4, iterations_4, grads_4, weights_4 = ex7_callback()
    input_layer = FullyConnectedLayer(n_features, 6, ReLU())
    hidden_layer_1 = FullyConnectedLayer(6, 6, ReLU())
    hidden_layer_2 = FullyConnectedLayer(6, n_classes, ReLU())
    layers = [input_layer, hidden_layer_1, hidden_layer_2]
    nn_4 = NeuralNetwork(layers, CrossEntropyLoss(),
                         GradientDescent(learning_rate=FixedLR(0.1),
                                         max_iter=5000,
                                         callback=callback))
    nn_4.fit(train_X, train_y)
    pred = nn_4.predict(test_X)

    fig_4 = go.Figure()
    fig_4.add_trace(go.Scatter(x=iterations_4, y=loss_4, mode="lines",
                               marker=dict(color="blue"), name="loss",
                               showlegend=True))
    fig_4.update_layout(
        title=f"Loss as Function of Iteration",
        xaxis_title="iteration",
        title_x=0.5)
    fig_4.show()

    fig_4 = go.Figure()
    fig_4.add_trace(go.Scatter(x=iterations_4, y=grads_4, mode="lines",
                               marker=dict(color="red"), name="gradient norm",
                               showlegend=True))
    fig_4.update_layout(
        title=f"Gradient Norm as Function of Iteration",
        xaxis_title="iteration",
        title_x=0.5)
    fig_4.show()


