from typing import List

import joblib
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from enum import Enum
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import mode
from sklearn.metrics import classification_report, \
    accuracy_score, recall_score, f1_score, precision_score


def calc_metrics(y, preds, round_b=False, num_classes=2, average='macro'):
    accuracy = accuracy_score(y, preds)

    # we use unweighted average so that all classes get considered equally
    uar = recall_score(y, preds, labels=np.arange(num_classes), average=average)
    prec = precision_score(y, preds, labels=np.arange(num_classes), average=average)
    fscore = f1_score(y, preds, average="macro")

    # get scores for each class
    recall_per_class = recall_score(y, preds, labels=np.arange(num_classes), average=None)
    prec_per_class = precision_score(y, preds, labels=np.arange(num_classes), average=None)
    fscore_per_class = f1_score(y, preds, labels=np.arange(num_classes), average=None)

    if round_b:
        return round(accuracy, 2), round(uar, 2), round(prec, 2), round(fscore,
                                                                        2), recall_per_class, prec_per_class, fscore_per_class
    else:
        return accuracy, uar, prec, fscore, recall_per_class, prec_per_class, fscore_per_class


def cut_data_into_windows(X_data: np.ndarray, Y_data: np.ndarray, window_length: int, overlap: int, train: bool):
    """
    very basic windowing function
    """
    assert len(X_data.shape) == 2, f"Expected X_data to have a shape of 2 but it was \"{len(X_data.shape)}\""
    # only use overlap on train data
    if not train:
        overlap = 0

    if window_length >= len(X_data):
        pass
        # TODO: figure out why some videos are very short (<0.8s) and then put assertion back in
    assert window_length < len(X_data), "The window length exceeds the length of the data"

    sequences = []
    position = 0
    label_sequences = []
    while position < len(X_data):

        if position + window_length >= len(X_data):
            sample = X_data[position:-1]
            label_sample = Y_data[position:-1]
        else:
            sample = X_data[position:position + window_length]
            label_sample = Y_data[position:position + window_length]

        position = position + window_length - overlap

        label_sequences.append(label_sample)
        sequences.append(sample)

    # NULL PADDING
    sequences = pad_sequences(sequences, dtype='float32', maxlen=window_length, padding='post')
    label_sequences = pad_sequences(label_sequences, dtype="float32", maxlen=window_length, padding="post")
    labels_out = []

    for seq in label_sequences:
        labels_out.append(np.asarray([mode(seq.flatten()).mode[0]], dtype=int))

    labels_out = np.asarray(labels_out)
    return sequences, labels_out


class StatMetric(Enum):
    mean = "mean"
    std = "std"
    var = "var"
    median = "median"
    ptp = "ptp"  # peak to peak
    min = "min"
    max = "max"

    def __str__(self):
        return str(self.value)


def transform_sample_to_stat(X, metrics, feature_names=None):
    """
    (num_samples, frames, num_feat)

    1. sample

    1 2 3 4
    2 .
    3   .
    4     .
    ^ ^ ^ ^
    [avg, avg, avg, avg] -> (num_feat, )

    """
    # Calculate stats for each column and create new features (avg, std, ... for all features)

    if feature_names is None:
        feature_names = []

    X_ = []
    for x in X:
        stats = []

        for m in metrics:
            if m == StatMetric.std:
                stats.append(np.std(x, axis=0))
            elif m == StatMetric.mean:
                stats.append(np.mean(x, axis=0))
            elif m == StatMetric.var:
                stats.append(np.var(x, axis=0))
            elif m == StatMetric.ptp:
                stats.append(np.ptp(x, axis=0))
            elif m == StatMetric.median:
                stats.append(np.median(x, axis=0))
            elif m == StatMetric.min:
                stats.append(np.min(x, axis=0))
            elif m == StatMetric.max:
                stats.append(np.max(x, axis=0))

        x_new = None
        for s in stats:
            if x_new is None:
                x_new = np.copy(s)
            else:
                x_new = np.hstack((x_new, s))

        X_.append(x_new)

    # create new feature labels

    labels = []
    for m in metrics:
        for l in feature_names:
            labels.append(str(l) + "_" + str(m))

    return np.array(X_), labels


# def remove_nan_values(a: np.ndarray) -> np.ndarray:
#     if len(a.shape) == 1:
#         return a[~np.isnan(a).any()]
#     return a[~np.isnan(a).any(axis=1)]


def standardize_data(scaler: StandardScaler, X_data: np.ndarray,
                     train: bool = False, model_name: str = None,
                     results_dir: str = "results"):
    """
    Standardize features by removing the mean and scaling to unit variance.
    X: shape = (num_samples, num_frames, num_features)
    """

    if train:
        fitted_scaler = scaler.fit(X_data.reshape(-1, X_data.shape[-1]))
        joblib.dump(fitted_scaler,
                    os.path.join(results_dir,
                                 "fitted_scaler" if model_name is None else f"fitted_scaler_{model_name}"),
                    compress=True)

    fitted_scaler = scaler

    X_data_shape = X_data.shape

    X_data = fitted_scaler.transform(X_data.reshape(-1, X_data.shape[-1]))

    ret = X_data.reshape(X_data_shape)

    return ret


def concat_dfs(dfs: List[pd.DataFrame], idxes: List[int]) -> pd.DataFrame:
    assert len(dfs) > max(idxes)

    if len(idxes) == 1:
        return dfs[idxes[0]]
    res = []

    for idx in idxes:
        res.append(dfs[idx])

    return pd.concat(res,
                     axis=0)


def next_cross_validation_split(dataset_dfs: List[pd.DataFrame], limit: int = None):
    assert len(dataset_dfs) > 2
    all_idxes = set(range(len(dataset_dfs)))
    counter = 0
    for test_idx in range(len(dataset_dfs)):
        for valid_idx in range(len(dataset_dfs)):
            if test_idx == valid_idx:
                continue

            train_idxes = list(all_idxes - {test_idx, valid_idx})

            train_df = concat_dfs(dataset_dfs, train_idxes)
            test_df = concat_dfs(dataset_dfs, [test_idx])
            validation_df = concat_dfs(dataset_dfs, [valid_idx])

            yield train_df, test_df, validation_df

            counter += 1

            if limit is not None:
                if counter >= limit:
                    return
