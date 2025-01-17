from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    observations = pd.read_csv(filename)
    # delete non relevant columns of data
    observations.drop(columns=['id', 'date', 'long', 'lat'], inplace=True)
    # remove samples where price is <= zero
    observations["neg_vals"] = (observations["price"] > 0)
    to_drop = []
    for row in range(len(observations["neg_vals"])):
        if not observations["neg_vals"][row]:
            to_drop.append(row)
    observations.drop(to_drop, inplace=True)
    observations.drop(columns=["neg_vals"], inplace=True)
    prices = observations.pop('price')
    return observations, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature_name, feature in X.iteritems():
        # calculate pearson corrolation:
        p = (np.cov(feature, y) / (np.std(feature) * np.std(y)))[1][0]
        fig = px.scatter(x=feature, y=y, labels={"x": feature_name, "y": "Price"},
                         title="Price as a function of " + str(feature_name) +
                               ". Pearson Corrolation = " + str(round(p, 3)))
        fig.write_image(output_path + str(feature_name) + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Loading and preprocessing of housing prices dataset
    X, y = load_data("C:\\Users\\97250\\PycharmProjects\\IML.HUJI\\datasets\\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "C:\\Users\\97250\\PycharmProjects\\IML.HUJI\\exercises\\plots_house_prices\\")

    # Question 3 - Split samples into training and testing sets.

    # continue pre-processing based on results of question 2:
    # convert zipcode to dummy values since their value doesn't have logical order
    X = pd.get_dummies(X, columns=['zipcode'])
    # drop yr_built column since it is not beneficial to the model
    X.drop(columns=['yr_built'], inplace=True)

    X_train, y_train, X_test, y_test = split_train_test(X, y)
    # print(len(X_train), len(X_test))

    # Question 4 - Fit model over increasing percentages of the overall training data
    lin_reg = LinearRegression()
    percent = []
    loss = []
    std_loss = []

    for i in range(10, 101):
        p = i / 100
        # list for the 10 values of lost function we will compute, for the current p value
        ten_loss = []
        for j in range(10):
            split = np.random.rand(X_train.shape[0]) > p
            curr_X_train = X_train[~split].to_numpy()
            curr_y_train = y_train[~split].to_numpy()
            lin_reg.fit(curr_X_train, curr_y_train)
            ten_loss.append(lin_reg.loss(X_test.to_numpy(), y_test.to_numpy()))
        avg_loss = sum(ten_loss) / 10
        percent.append(p)
        loss.append(avg_loss)
        # calculate the std of the 10 fits we have made with current value of p
        std_loss.append(np.std(ten_loss))
    std_loss = 2 * np.array(std_loss)
    loss = np.array(loss)
    confidence_plus = loss + std_loss
    confidence_minus = loss - std_loss
    # print(confidence_minus.shape, confidence_plus.shape, loss.shape)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percent, y=loss, mode="markers+lines",
                              marker=dict(color="blue"), showlegend=False))
    fig.add_traces([
        go.Scatter(x=percent, y=confidence_plus, fill="tonexty",
                   line=dict(color="lightgrey"), showlegend=False),
        go.Scatter(x=percent, y=confidence_minus, fill="tonexty",
                   line=dict(color="lightgrey"),showlegend=False)
    ])

    fig.update_layout(title="Mean MSE Loss as Function of Percent of Samples",
                      xaxis_title="% Of Training Sample", yaxis_title="MSE Loss")
    fig.show()

    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
