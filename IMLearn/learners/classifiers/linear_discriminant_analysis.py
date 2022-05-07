from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier
    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`
    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`
    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`
    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`
    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.
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
        # mu
        mu = []
        for i in self.classes_:
            mu.append(X[y == i].mean(axis=0))
        self.mu_ = np.array(mu)
        # cov

        d = dict()  # maps class name to index
        for index, value in enumerate(self.classes_):
            d[value] = index


        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            self.cov_ += np.outer((X[i] - self.mu_[d[y[i]]]), (X[i] - self.mu_[d[y[i]]]))
        self.cov_ /= (X.shape[0] - len(self.classes_))

        # inv
        self._cov_inv = np.linalg.inv(self.cov_)
        self.fitted_ = True
        # print("classes:")
        # print(self.classes_)
        # print("mu:")
        # print(self.mu_)
        # print("cov:")
        # print(self.cov_)
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

        def prob(x, k):
            b_k = np.log(self.pi_[k]) - 0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k]
            a_k = self._cov_inv @ self.mu_[k]
            return a_k.T @ x + b_k

        likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(X.shape[0]):
            for j in range(len(self.classes_)):
                likelihood[i, j] = prob(X[i], j)

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
