import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    observations = pd.read_csv(filename, parse_dates={'DayOfYear': [2]})
    observations = observations.loc[observations.Temp > -50]
    observations['DayOfYear'] = observations['DayOfYear'].apply(lambda x: x.dayofyear)
    # print(observations.head())
    return observations


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    all_data = load_data('C:\\Users\\97250\\PycharmProjects\\IML.HUJI\\datasets\\City_Temperature.csv')
    # Question 2 - Exploring data for specific country
    israel_data = all_data.loc[all_data.Country == 'Israel']
    # 2.a
    fig = px.scatter(x=israel_data['DayOfYear'], y=israel_data['Temp'],
                     labels={"x": "Day Of Year", "y": "Temperature in Celsius"},
                     color=israel_data['Year'].apply(lambda x: str(x)),
                     title="Temperatures as function of Day of Year in Tel Aviv, Israel ")

    fig.show()

    # 2.b
    group_by_month = israel_data.groupby('Month').std()
    fig = px.bar(x=[i for i in range(1,13)], y=group_by_month['Temp'],
                     labels={"x": "Month", "y": "Standard Deviation of Daily Temperatures"},
                     title="Standard Deviation of Daily Temperatures Each Month ")

    fig.show()

    # Question 3 - Exploring differences between countries

    group_by_month_country = all_data.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']})
    group_by_month_country.columns = ['Mean Temp', 'STD']
    group_by_month_country = group_by_month_country.reset_index()

    fig = px.line(group_by_month_country, x='Month', y='Mean Temp', error_y='STD', color='Country',
                  title="Mean Temperatures of Countries by Month")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    y = israel_data.pop('Temp')
    X = israel_data['DayOfYear']
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    pol_deg = []
    loss = []
    for k in range(1, 11):  # todo inheritance
        pol_regg = PolynomialFitting(k)
        pol_regg.fit(X_train, y_train)
        pol_deg.append(k)
        loss.append(round(pol_regg.loss(X_test.to_numpy(), y_test.to_numpy()), 2))
    print(pol_deg)
    print(loss)
    fig = px.bar(x=pol_deg, y=loss, labels={"x": "Polynom Degree", "y": "MSE Loss"},
                     title="MSE Loss as function of Polynom Degree")

    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    country = []
    loss_country = []

    israel_fit = PolynomialFitting(6)
    israel_fit.fit(X, y)

    jordan_data = all_data.loc[all_data.Country == 'Jordan']
    y = jordan_data.pop('Temp')
    X = jordan_data['DayOfYear']
    country.append('Jordan')
    loss_country.append(israel_fit.loss(X, y))

    africa_data = all_data.loc[all_data.Country == 'South Africa']
    y = africa_data.pop('Temp')
    X = africa_data['DayOfYear']
    country.append('South Africa')
    loss_country.append(israel_fit.loss(X, y))

    holland_data = all_data.loc[all_data.Country == 'The Netherlands']
    y = holland_data.pop('Temp')
    X = holland_data['DayOfYear']
    country.append('The Netherlands')
    loss_country.append(israel_fit.loss(X, y))

    fig = px.bar(x=country, y=loss_country, labels={"x": "Country", "y": "MSE Loss"},
                     title="MSE Loss for each country, by Polyfit of Deg 6 on Israel Data")

    fig.show()




