import time as ti
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


# callback function for question 10
def time_callback():
    time_record = []
    loss = []

    def callback(**kwargs):
        time_record.append(ti.time())
        loss.append(np.mean(kwargs["val"]))

    return callback, time_record, loss


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    from nn_simulated_data import ex7_callback

    callback, loss_1, iterations_1, grads_1, weights_1 = ex7_callback()
    input_layer = FullyConnectedLayer(train_X.shape[1], 64, ReLU())
    hidden_layer_1 = FullyConnectedLayer(64, 64, ReLU())
    hidden_layer_2 = FullyConnectedLayer(64, n_classes)
    layers = [input_layer, hidden_layer_1, hidden_layer_2]
    nn = NeuralNetwork(layers, CrossEntropyLoss(),
                       StochasticGradientDescent(learning_rate=FixedLR(0.1),
                                                 max_iter=10000,
                                                 batch_size=256,
                                                 callback=callback))
    nn.fit(train_X, train_y)
    pred = nn.predict(test_X)
    print(f"Question 6: acuracy = {accuracy(test_y, pred)}")

    # Plotting convergence process
    fig_6 = go.Figure()
    fig_6.add_trace(go.Scatter(x=iterations_1, y=loss_1, mode="lines",
                               marker=dict(color="blue"), name="loss",
                               showlegend=True))
    fig_6.update_layout(
        title=f"Loss as function of iteration",
        xaxis_title="iteration",
        yaxis_title="loss",
        title_x=0.5)
    fig_6.show()

    fig_6 = go.Figure()
    fig_6.add_trace(go.Scatter(x=iterations_1, y=grads_1, mode="lines",
                               marker=dict(color="red"), name="Gradient Norm",
                               showlegend=True))
    fig_6.update_layout(
        title=f"Gradient Norm as function of iteration",
        xaxis_title="iteration",
        yaxis_title="Gradient Norm",
        title_x=0.5)
    fig_6.show()

    # Plotting test true- vs predicted confusion matrix

    conf_mat = confusion_matrix(test_y, pred)
    print(conf_mat)

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#

    callback, loss_2, iterations_2, grads_2, weights_2 = ex7_callback()
    input_layer = FullyConnectedLayer(train_X.shape[1], 64)
    layers = [input_layer]
    nn_no_layers = NeuralNetwork(layers, CrossEntropyLoss(),
                                 StochasticGradientDescent(learning_rate=FixedLR(0.1),
                                                           max_iter=10000,
                                                           batch_size=256,
                                                           callback=callback))
    nn_no_layers.fit(train_X, train_y)
    pred = nn_no_layers.predict(test_X)
    print(f"Question 8: acuracy = {accuracy(test_y, pred)}")

    # Plotting convergence process
    fig_8 = go.Figure()
    fig_8.add_trace(go.Scatter(x=iterations_2, y=loss_2, mode="lines",
                               marker=dict(color="blue"), name="loss",
                               showlegend=True))
    fig_8.update_layout(
        title=f"Loss as function of iteration",
        xaxis_title="iteration",
        yaxis_title="loss",
        title_x=0.5)
    fig_8.show()

    fig_8 = go.Figure()
    fig_8.add_trace(go.Scatter(x=iterations_2, y=grads_2, mode="lines",
                               marker=dict(color="red"), name="Gradient Norm",
                               showlegend=True))
    fig_8.update_layout(
        title=f"Gradient Norm as function of iteration",
        xaxis_title="iteration",
        yaxis_title="Gradient Norm",
        title_x=0.5)
    fig_8.show()
    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    train_X_7 = train_X[train_y == 7]
    train_y_7 = train_y[train_y == 7]
    test_X_7 = test_X[test_y == 7]
    test_y_7 = test_y[test_y == 7]
    nn.fit(train_X_7, train_y_7)
    nn.predict(test_X_7)
    # extract from the network our probability vector
    pred_probs = nn.probs
    pred_with_X = np.concatenate((pred_probs, test_X_7), axis=1)
    sorted_pred = pred_with_X[pred_with_X[:, 0].argsort()]

    most_confident = sorted_pred[:64, 10:]
    least_confident = sorted_pred[-64:, 10:]
    print(least_confident.shape)
    print(most_confident.shape)

    confident_plot = plot_images_grid(np.array(most_confident), "most confident")
    not_confident_plot = plot_images_grid(np.array(least_confident), "least confident")

    confident_plot.show()
    not_confident_plot.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#

    train_X = train_X[:2500]
    train_y = train_y[:2500]

    callback_SGD, time_SGD, loss_SGD = time_callback()
    input_layer_SGD = FullyConnectedLayer(train_X.shape[1], 64, ReLU())
    hidden_layer_1_SGD = FullyConnectedLayer(64, 64, ReLU())
    hidden_layer_2_SGD = FullyConnectedLayer(64, n_classes)
    layers = [input_layer_SGD, hidden_layer_1_SGD, hidden_layer_2_SGD]
    nn_SGD = NeuralNetwork(layers, CrossEntropyLoss(),
                           StochasticGradientDescent(learning_rate=FixedLR(0.1),
                                                     max_iter=10000,
                                                     batch_size=256,
                                                     tol=10e-10,
                                                     callback=callback_SGD))

    nn_SGD.fit(train_X, train_y)
    sgd_time_final = [int(i - time_SGD[0]) for i in time_SGD]
    print(sgd_time_final)
    fig_10 = go.Figure()
    fig_10.add_trace(go.Scatter(x=sgd_time_final, y=loss_SGD, mode="lines",
                                marker=dict(color="blue"), name="loss_SGD",
                                showlegend=True))
    fig_10.update_layout(
        title=f"Loss as function of Time",
        xaxis_title="time",
        yaxis_title="loss",
        title_x=0.5)
    fig_10.show()

    callback_GD, time_GD, loss_GD = time_callback()
    input_layer_GD = FullyConnectedLayer(train_X.shape[1], 64, ReLU())
    hidden_layer_1_GD = FullyConnectedLayer(64, 64, ReLU())
    hidden_layer_2_GD = FullyConnectedLayer(64, n_classes)
    layers = [input_layer_GD, hidden_layer_1_GD, hidden_layer_2_GD]
    nn_GD = NeuralNetwork(layers, CrossEntropyLoss(),
                          GradientDescent(learning_rate=FixedLR(0.1),
                                          max_iter=10000,
                                          tol=10e-10,
                                          callback=callback_GD))
    nn_GD.fit(train_X, train_y)
    gd_time_final = [int(i - time_GD[0]) for i in time_GD]
    print(gd_time_final)

    fig_11 = go.Figure()
    fig_11.add_trace(go.Scatter(x=gd_time_final, y=loss_GD, mode="lines",
                                marker=dict(color="blue"), name="loss_GD",
                                showlegend=True))
    fig_11.update_layout(
        title=f"Loss as function of Time",
        xaxis_title="time",
        yaxis_title="loss",
        title_x=0.5)
    fig_11.show()

    # both models on single plot as requested
    fig_11.add_trace(go.Scatter(x=gd_time_final, y=loss_SGD, mode="lines",
                                marker=dict(color="red"), name="loss_SGD",
                                showlegend=True))
    fig_11.show()
