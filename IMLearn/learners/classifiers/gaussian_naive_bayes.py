from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier
        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`
        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`
        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`
        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # classes & pi
        unique, counts = np.unique(y, return_counts=True)
        self.pi_ = np.array([i / X.shape[0] for i in counts])
        self.classes_ = unique
        # mu & vars
        mu = []
        var = []
        for i in self.classes_:
            mu.append(X[y == i].mean(axis=0))
            var.append(X[y == i].var(axis=0, ddof=1))
        self.mu_ = np.array(mu)
        self.vars_ = np.array(var)
        # print("classes:")
        # print(self.classes_)
        print("mu:")
        print(self.mu_)
        print("vars:")
        print(self.vars_)
        # print("pi:")
        # print(self.pi_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        predictions = [0] * X.shape[0]
        likelihoods = self.likelihood(X)
        for i in range(likelihoods.shape[0]):
            current_max = None
            # find the best class for the current feature
            for k in range(likelihoods.shape[1]):
                if current_max is None or likelihoods[i, k] > current_max:
                    current_max = likelihoods[i, k]
                    predictions[i] = self.classes_[k]
        return predictions

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.
        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # log likelihood:

        def prob(x, k):
            ans = 0
            for feature in range(len(x)):
                ans += np.log(1 / (np.sqrt(self.vars_[k, feature] * 2 * np.pi))) \
                       - 0.5 * ((x[feature] - self.mu_[k, feature]) / np.sqrt(self.vars_[k, feature])) ** 2
            return ans

        likelihood = np.zeros((X.shape[0], len(self.classes_)))

        for sample in range(X.shape[0]):
            for clss in range(len(self.classes_)):
                likelihood[sample, clss] = np.log(self.pi_[clss]) + prob(X[sample], clss)

        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))