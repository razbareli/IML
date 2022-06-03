import os

import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import Data.evaluate_part_0 as evaluate_file_0
import Data.evaluate_part_1 as evaluate_file_1

ind_to_label = dict()
label_to_ind = dict()

def her2_column(data, col_name):
    neg_regex = '[NnGg]|[של]|0'
    pos_regex = '[PpSs]|[חב]|[123]|-|\+'
    data[col_name].replace(pos_regex, 1, regex=True, inplace=True)
    data[col_name].replace(neg_regex, 0, regex=True, inplace=True)
    data[col_name].fillna(0, inplace=True)
    temp = (data[col_name] == 0) | (data[col_name] == 1)
    data = data[temp]
    return data


def make_column_timestamp(data_frame, col_name):
    data_frame[col_name] = \
        data_frame[col_name].astype(str).str[0:10]

    data_frame[col_name] = \
        data_frame[col_name].replace('Unknown', np.NaN)

    data_frame[col_name] = \
        data_frame[col_name].astype('datetime64[ns]')

    data_frame[col_name] = data_frame[col_name].values.astype(np.int64) // 10 ** 9

def create_string_labeled_data(predictions):
    converted = []
    for line in predictions:
        curr = []
        for i in range(len(line)):
            if line[i] == 1:
                curr.append(ind_to_label[i])
        converted.append(str(curr))
    return converted


def create_multi_hot_labels(labels):
    """
    Turns the string labels into multi hot/
    """
    classes = []
    data = []
    for row in labels:
        s = str(row).strip("[").strip("]").replace("'", "").split(", ")
        data.append(s)
        for i in s:
            if len(i) > 0 and i not in classes:
                ind_to_label[len(classes)] = i
                label_to_ind[i] = len(classes)
                classes.append(i)
    m = MultiLabelBinarizer(classes=classes)
    m.fit(data)
    return m.transform(data)


def parse():
    # Use a breakpoint in the code line below to debug your script.
    # todo take the df from the args and not like that
    data_frame = pd.read_csv('train.feats.csv')
    labels_0 = pd.read_csv("train.labels.0.csv")
    labels_1 = pd.read_csv("train.labels.1.csv")
    # This is done so that the rows that are removed will be for the labels too
    data_frame = pd.concat(objs=[labels_0, labels_1, data_frame], axis=1)

    data_frame.drop(
        columns=[
            ' Form Name',
            ' Hospital',
            'User Name',
            'אבחנה-Ivi -Lymphovascular invasion',
            'אבחנה-KI67 protein',
            'אבחנה-Side',
            'אבחנה-Stage',

            'אבחנה-Surgery date1',
            'אבחנה-Surgery name1',
            'אבחנה-Surgery date2',
            'אבחנה-Surgery name2',
            'אבחנה-Surgery date3',
            'אבחנה-Surgery name3',

            'אבחנה-Tumor depth',
            'אבחנה-Tumor width',
            'אבחנה-Surgery sum',

            'אבחנה-er',
            'אבחנה-pr',

            'surgery before or after-Activity date',
            'surgery before or after-Actual activity',
            'id-hushed_internalpatientid',

            'אבחנה-N -lymph nodes mark (TNM)'
        ], inplace=True)

    # data_frame["decade_born"] = (data_frame["אבחנה-Age"] / 1).astype(int)
    # data_frame = pd.get_dummies(data_frame, columns=["decade_born"], drop_first=True)
    # data_frame.drop(['אבחנה-Age'], inplace=True, axis=1)

    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Basic stage'], drop_first=True)
    data_frame = her2_column(data_frame, 'אבחנה-Her2')
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Her2'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Histological diagnosis'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Histopatological degree'], drop_first=True)

    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Lymphatic penetration'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-M -metastases mark (TNM)'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Margin Type'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-T -Tumor mark (TNM)'], drop_first=True)

    data_frame['אבחנה-Nodes exam'] = data_frame['אבחנה-Nodes exam'].fillna(0)
    data_frame['אבחנה-Positive nodes'] = data_frame['אבחנה-Positive nodes'].fillna(0)
    # data_frame["nodes_exam_pref"] = (data_frame['אבחנה-Nodes exam'].fillna(0) // 10).astype(int)
    # data_frame.drop(['אבחנה-Nodes exam'], inplace=True, axis=1)
    #
    # data_frame["pos_nodes_pref"] = (data_frame['אבחנה-Positive nodes'].fillna(0) // 10).astype(int)
    # data_frame.drop(['אבחנה-Positive nodes'], inplace=True, axis=1)

    make_column_timestamp(data_frame,'אבחנה-Diagnosis date')

    return data_frame



def predict_0(y, X):
    y = create_multi_hot_labels(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    tree = RandomForestClassifier()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_val)

    print("Evaluation: ", send_to_evaluation_0(y_val, y_pred))

def predict_1(y, X):
    # y = create_multi_hot_labels(y)
    # todo make the Tumor location thing into a multi hot feature too

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    tree = RandomForestRegressor()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)

    print("Evaluation: ", send_to_evaluation_1(y_train, y_pred))



def send_to_evaluation_0(y_gold, y_pred):
    """ Receives our multi-hot, puts it in a csv and evaluates"""
    y_gold_df = pd.DataFrame(create_string_labeled_data(y_gold))
    y_pred_df = pd.DataFrame(create_string_labeled_data(y_pred))
    y_gold_df.to_csv('temp_gold.labels.0.csv', index=False)
    y_pred_df.to_csv('temp_pred.labels.0.csv', index=False)
    macro_f1, micro_f1 = evaluate_file_0.evaluate('temp_gold.labels.0.csv',
                                                "temp_pred.labels.0.csv")
    os.remove('temp_gold.labels.0.csv')
    os.remove('temp_pred.labels.0.csv')
    return macro_f1, micro_f1

def send_to_evaluation_1(y_gold, y_pred):
    """ Receives our multi-hot, puts it in a csv and evaluates"""
    y_gold_df = pd.DataFrame(list(y_gold))
    y_pred_df = pd.DataFrame(list(y_pred))
    y_gold_df.to_csv('temp_gold.labels.1.csv', index=False)
    y_pred_df.to_csv('temp_pred.labels.1.csv', index=False)
    mse = evaluate_file_1.evaluate('temp_gold.labels.1.csv',
                                                "temp_pred.labels.1.csv")
    os.remove('temp_gold.labels.1.csv')
    os.remove('temp_pred.labels.1.csv')
    return mse


if __name__ == '__main__':
    np.random.seed(0)
    df = parse()
    y_0, X_0 = df["אבחנה-Location of distal metastases"], df.drop(
        columns=["אבחנה-Location of distal metastases"])
    y_1, X_1 = df["אבחנה-Tumor size"], df.drop(
        columns=["אבחנה-Tumor size"])

    # todo this should be replaced by multi hot:
    X_1 = X_1.drop(columns=["אבחנה-Location of distal metastases"])
    predict_0(y_0, X_0)
    predict_1(y_1, X_1)