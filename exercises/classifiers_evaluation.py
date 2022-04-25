from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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

        naive = GaussianNaiveBayes()
        naive.fit(X, y)
        naive_pred = naive.predict(X)



        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        lda_accuracy = accuracy(y, lda_pred)
        naive_accuracy = accuracy(y, naive_pred)

        # list just for iterating while plotting
        predictions = [None, lda_pred, naive_pred]

        from IMLearn.metrics import accuracy
        lda_accuracy = round(accuracy(y, lda_pred), 3)
        naive_accuracy = round(accuracy(y, naive_pred), 3)

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            "LDA Classifier <br> Accuracy = " + str(lda_accuracy),
            "Naive Base Gaussian Classifier <br> Accuracy = " + str(naive_accuracy)],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for index, estimator in enumerate([naive, lda], start=1):
            # Add the data
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=predictions[index], symbol=y,
                                                 line=dict(color="black", width=1))), row=1, col=index)

            # Add `X` for mean of distribution
            fig.add_trace(go.Scatter(x=estimator.mu_[:, 0], y=estimator.mu_[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color='black', symbol='x', size=12,
                                                 line=dict(color="black", width=1))), row=1, col=index)

        fig.update_layout(title_text=f"Classifier Comparison Dataset: {f}", title_x=0.5)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()


