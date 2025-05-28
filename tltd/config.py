
import pandas as pd
import numpy as np
NUM_FOLDS_OUTTER = 2

def data_loader():

    d0 = 'datasets/zhh_b.csv'
    data_names = [d0]
    data_frames = []
    for csv_name in data_names:
        temp_df = pd.read_csv(csv_name)
        temp_df = temp_df.set_axis([*temp_df.columns[:-1], 'class'], axis=1)

        temp_df = temp_df.fillna(temp_df.mean())
        for col_name in temp_df.columns:
            if temp_df[col_name].dtype == "object":
                temp_df[col_name] = pd.Categorical(temp_df[col_name])
                temp_df[col_name] = temp_df[col_name].cat.codes
        X = temp_df.drop('class', axis=1)
        y = temp_df['class']
        data_frames.append((X, y, len(pd.unique(temp_df['class']))))

    return data_frames


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def one_hot(y_test, n_class):
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1, 1)
    y_test = indices_to_one_hot(y_test, n_class)

    return y_test


class Data:
    pass

