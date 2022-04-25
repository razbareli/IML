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
        # lda = LDA()
        # lda.fit(X, y)
        # print(lda.likelihood(X))
        # print(lda.predict(X))
        # print(sum(lda.predict(X) == y))

        # test lda
        # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # test_lda = LinearDiscriminantAnalysis(store_covariance=True)
        # test_lda.fit(X, y)
        # print(test_lda.covariance_)
        # print(sum(test_lda.predict(X) == y))

        naive = GaussianNaiveBayes()
        X = np.array([[1,1],[1,2], [2,3], [2,4], [3,3], [3,4]])
        y = np.array([0,0,1,1,1,1])
        naive.fit(X, y)
        # print(naive.likelihood(X))
        # print(naive.predict(X))
        # print(sum(naive.predict(X) == y))

        # test naive
        # from sklearn.naive_bayes import GaussianNB
        # test_naive = GaussianNB()
        # test_naive.fit(X, y)
        # print(sum(test_naive.predict(X) == y))



        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    # compare_gaussian_classifiers()
    # arr = np.array([1,0,0,1,2,2,2,2,2,1,1,30,1,2,1,1])
    # print(np.where(np.amax(arr)))
    # print(arr == 0)
    # unique, counts = np.unique(arr, return_counts=True)
    # print(unique, counts)
    # pi = np.array([i/len(arr) for i in counts])
    # mat = np.array([[1,2,3],[2,3,3]])
    # print(mat[0])
    # vec = np.array([1,0])
    # print(mat[vec==1].mean(axis=0))

