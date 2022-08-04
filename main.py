import numpy as np

from utils import load_dataset, plot_df, plt
from preprocessing import next_cross_validation_split, \
    standardize_data, remove_nan_values, StatMetric, transform_sample_to_stat, \
    cut_data_into_windows, calc_metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import datetime
from lstm import LSTMModel, custom_lstm_model, default_callbacks
from sklearn.svm import OneClassSVM
import time
from resample import print_class_distribution
from scikit_pipelines import create_confusion_matrices_from_values
from tensorflow import keras


def create_sequences(df, scaler, model_name, results_dir, window_len: int, overlap: int, train: bool):
    X_data, Y_data = df.loc[:, df.columns != 'Label'].values, df["Label"].values
    X_data = standardize_data(scaler,
                              X_data,
                              train=True,
                              model_name=model_name,
                              results_dir=results_dir)
    X_data = remove_nan_values(X_data)
    Y_data = remove_nan_values(Y_data)
    sequences, labels = cut_data_into_windows(X_data, Y_data, window_len, overlap, train)
    return sequences, labels


def run():
    dataset_samples = load_dataset("ACSS")
    dataset_dfs = [x.single_df_without_time for x in dataset_samples]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    window_length = 30
    overlap = 0
    features_num = 9
    input_shape = (window_length, features_num)
    epochs = 10
    for train_df, test_df, validation_df in next_cross_validation_split(dataset_dfs, limit=1):
        model_name = str(datetime.datetime.now())

        scaler = StandardScaler()

        train_sequences, train_labels = create_sequences(train_df, scaler, model_name, results_dir,
                                                         window_length, overlap, True)

        validation_sequences, validation_labels = create_sequences(validation_df,
                                                                   scaler, model_name, results_dir,
                                                                   window_length, overlap, False)
        test_sequences, test_labels = create_sequences(test_df,
                                                       scaler, model_name, results_dir,
                                                       window_length, overlap, False)

        print_class_distribution(train_labels)
        print_class_distribution(validation_labels)

        lstm_model = LSTMModel({}, custom_lstm_model(input_shape))
        lstm_model.fit(train_sequences, train_labels, validation_sequences,
                       validation_labels,
                       epochs=epochs,
                       callbacks=default_callbacks(results_dir, model_name)
                       )
        # lstm_model = keras.models.load_model('resultsmodel_2022-08-04 17:50:32.313217.h5')
        predictions = lstm_model.predict(test_sequences)
        predictions = np.apply_along_axis(lambda x: int(x > 0.50), 1, predictions)
        test_labels = test_labels.flatten()
        predictions[0] = 0
        test_labels[0] = 0
        create_confusion_matrices_from_values(true_values=[test_labels],
                                              predicted_values=[predictions], plot_names=[model_name],
                                              display_labels=["RUNNING","CUT"],
                                              show=True,
                                              # normalize=None
                                              )


if __name__ == "__main__":
    run()
