from sklearn.metrics import accuracy_score, roc_auc_score

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    print(filename)
    full_data = pd.read_csv(filename).fillna(False)

    # pre-processing
    # convert data to numeric
    full_data.accommadation_type_name = full_data.accommadation_type_name.astype('category')
    full_data.customer_nationality = full_data.customer_nationality.astype(str).astype('category')
    full_data.hotel_id = full_data.hotel_id.astype('category')
    full_data.charge_option = full_data.charge_option.astype('category')
    full_data.cancellation_policy_code = full_data.cancellation_policy_code.astype('category')
    full_data['customer_nationality_codes'] = full_data.customer_nationality.cat.codes
    full_data['accommadation_type_name_codes'] = full_data.accommadation_type_name.cat.codes
    full_data['charge_option_codes'] = full_data.charge_option.cat.codes
    full_data['cancellation_policy_code_codes'] = full_data.cancellation_policy_code.cat.codes
    b2 = pd.to_datetime(full_data['checkin_date']).apply(lambda x: x.date())
    b1 = pd.to_datetime(full_data['booking_datetime']).apply(lambda x: x.date())
    full_data['days_before_checkin'] = [i.days for i in (b2.to_numpy() - b1.to_numpy())]

    full_data['days_before_checkin_pol'] = full_data["cancellation_policy_code"].str.extract(pat='^([0-9]*D)')
    full_data['days_before_checkin_pol'] = full_data['days_before_checkin_pol'].str[:-1]
    full_data['days_before_checkin_pol'] = full_data['days_before_checkin_pol'].fillna(0)
    pd.to_numeric(full_data['days_before_checkin'], errors='coerce')
    pd.to_numeric(full_data['days_before_checkin_pol'], errors='coerce')
    full_data['will_pay_for_cancellation'] = pd.to_numeric(full_data['days_before_checkin'], errors='coerce').subtract(
        pd.to_numeric(full_data['days_before_checkin_pol'], errors='coerce'), fill_value=0)
    full_data['will_pay_for_cancellation'] = full_data.will_pay_for_cancellation.map(lambda x: x > 0)
    # choose what data we will train the model on
    feature_cols = ["hotel_id",
                    "days_before_checkin",
                    "accommadation_type_name_codes",
                    "charge_option_codes",
                    "customer_nationality_codes",
                    "cancellation_policy_code_codes",
                    "will_pay_for_cancellation"
                    ]
    features = full_data[feature_cols]

    # full_data['label'] = full_data.cancellation_datetime.map(lambda x: bool(x) & True)
    # labels = full_data['label']

    # how many days after booking, the order was cancelled
    if filename.endswith("train.csv"):
        c1 = full_data['cancellation_datetime'].apply(lambda x: pd.Timestamp(x) if x is not False else x)
        c2 = full_data['booking_datetime'].apply(lambda x: pd.Timestamp(x))
        full_data['label2'] = [True if (c1[i] is not False and
                                        abs((c1[i] - c2[i]).days) <= 44) else False for i in range(len(c2))]
        labels2 = full_data['label2']
    else:
        labels2 = None
    return features, labels2


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)
    estimate = estimator.predict(X)
    return estimate


def f1_score(true: str, pred):
    true = pd.read_csv(true).fillna(False)
    labels = true['cancel']
    # print(labels)
    return classification_report(labels, pred, digits=4)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df_train, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    test_X, test_y, train_X, train_y, = split_train_test(df_train, cancellation_labels)
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    # load the real database
    df_pred, cancellation_labels = load_data(f"../datasets/test_set_week_10.csv")
    # Store model predictions over test set
    prediction = evaluate_and_export(estimator, df_pred, "209019256_203488747_204859326.csv")

    # check performance of previous weeks
    for w in range(1, 11):
        print(f"------------------------------ WEEK {w} --------------------------------- ")
        df_pred, cancellation_labels = load_data(f"../datasets/test_set_week_{w}.csv")
        # Store model predictions over test set
        prediction = evaluate_and_export(estimator, df_pred, "209019256_203488747_204859326.csv")
        # print(["False", "True"])
        # print(np.bincount(prediction))
        print(f1_score(f"../datasets/week_{w}_labels.csv", prediction))
        print(f"----------------------------------------------------------------------- ")

    # print("actual orders cancelled: ", np.bincount(test_y)[1])
    # print("predicted cancelled orders: ",np.bincount(prediction & test_y)[1])
    # print("correct guesses =  ", (np.bincount(prediction & test_y)[1]/np.bincount(test_y)[1])*100, "%")
