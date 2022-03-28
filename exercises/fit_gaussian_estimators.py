from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni = UnivariateGaussian()
    samples = np.random.normal(10, 1, 1000)
    uni.fit(samples)
    print("Expected Mean and Variance: \n", uni.mu_, uni.var_)

    # Question 2 - Empirically showing sample mean is consistent
    uni_partial = UnivariateGaussian()
    mean_diff = [0] * 100
    for i in range(1, 101):
        uni_partial.fit(samples[:i * 10])
        mean_diff[i - 1] = abs(uni_partial.mu_ - 10)
    fig = px.scatter(x=list([i * 10 for i in range(1, 101)]),
                     y=list(mean_diff), labels={"x": "Number of Samples",
                                                "y": "ABS of Difference"})
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    fig = px.scatter(x=samples, y=uni.pdf(samples), labels={"x": "Sample Value", "y": "Probabilyty"})
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean, cov, 1000)
    multi = MultivariateGaussian()
    multi.fit(samples)
    print("Expected Mean: \n", multi.mu_)
    print("Expected Covariance: \n", multi.cov_)

    # Question 5 - Likelihood evaluation
    likelihoods = np.zeros((200, 200))
    axis = np.linspace(-10, 10, 200)
    max_like = [None, None]  # maximum likelihood and its mu vector
    for i in range(len(axis)):
        for j in range(len(axis)):
            mu = np.array([axis[i], 0, axis[j], 0])
            likelihoods[i][j] = MultivariateGaussian.log_likelihood(mu, cov, samples)
            if max_like[0] is None or likelihoods[i][j] > max_like[0]:
                max_like[0] = likelihoods[i][j]
                max_like[1] = mu
    fig = px.imshow(likelihoods, labels=dict(x="f3", y='f1', color="log-likelihood"), x=axis, y=axis)
    fig.show()

    # Question 6 - Maximum likelihood
    print("max likelihood, corresponding mu vector:")
    print(max_like[0], max_like[1])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
