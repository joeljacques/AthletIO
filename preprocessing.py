from typing import List

import joblib
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

    return X_data.reshape(X_data_shape)


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
