from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    X = data[:, 0:2]
    y = data[:, 2]
    return X, y


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        callback_func = lambda fit, X_dummy, y_dummy: losses.append(fit.loss(X, y))
        perc = Perceptron(callback=callback_func)
        perc.fit(X, y)

        # Plot figure
        fig = go.Figure([go.Scatter(y=losses, mode='lines')],
                  layout=go.Layout(title=f"Loss as function of iteration, for {n} data",
                                   xaxis=dict(title="number of iterations"), yaxis=dict(title="losses")))
        fig.show()
def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)
        X = np.array([[1,1],[1,2],[2,3],[2,4],[3,3],[3,4]])
        y = np.array([0,0,1,1,1,1])
        naive = GaussianNaiveBayes()
        naive.fit(X, y)
        naive_pred = naive.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # from IMLearn.metrics import accuracy
        # lda_accuracy = accuracy(y, lda_pred)
        # naive_accuracy = accuracy(y, naive_pred)
        #
        # from IMLearn.metrics import accuracy
        # lda_accuracy = round(accuracy(y, lda_pred), 3)
        # naive_accuracy = round(accuracy(y, naive_pred), 3)
        #
        # fig = make_subplots(rows=1, cols=2, subplot_titles=[
        #     "Naive Base Gaussian Classifier <br> Accuracy = " + str(naive_accuracy),
        #     "LDA Classifier <br> Accuracy = " + str(lda_accuracy)],
        #                     horizontal_spacing=0.01, vertical_spacing=.03)
        # for index, model in enumerate([naive, lda], start=1):
        #     # add the data
        #     fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
        #                              marker=dict(color=model.predict(X), symbol=y,
        #                              line=dict(color="black", width=1))), row=1, col=index)
        #
        #     # add X symbol for mean of distribution
        #     fig.add_trace(go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1], mode="markers", showlegend=False,
        #                              marker=dict(color='black', symbol='x', size=12,
        #                              line=dict(color="black", width=1))), row=1, col=index)
        #
        #     # Add ellipses of covariances of fitted gaussians
        #     for k in range(len(model.classes_)):
        #         if model is lda:
        #             cov = model.cov_
        #         else:
        #             cov = np.diag(model.vars_[k])
        #         fig.add_trace(get_ellipse(model.mu_[k], cov), row=1, col=index)
        #
        # fig.update_layout(title_text=f"Classifier Comparison of {f}", title_x=0.5)
        # fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()


