from __future__ import annotations

import datetime
import os
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from lstm import LSTMModelWrapper
from preprocessing import next_cross_validation_split, \
    standardize_data, cut_data_into_windows
from resample import print_class_distribution
from scikit_pipelines import create_confusion_matrices_from_values, \
    create_pdf_from_figures, Path, calc_metrics
from utils import load_dataset, get_evaluation_results
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.utils.np_utils
import tensorflow as tf
import json


def default_callbacks(result_dir: str, model_name: str):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(result_dir, f"model_{model_name}.h5"),
                        save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.06,
                          patience=4, min_lr=0.00001)
    ]
    return callbacks


def custom_lstm_model(input_shape, model_name: str | None, learning_rate=0.0001):
    """
    input_shape: Input shape of the model

    Custom model function which can be passed as parameter to the pipeline
    If it is to be used in sklearn gridsearch/randomsearch additional
    hyperparameter can be passed as arguments
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=False,
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.L2(0.0001))(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("tanh")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name=model_name)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"]
                  )
    return model


def create_pdfs(models_and_data: List[Tuple[LSTMModelWrapper, str, np.ndarray, np.ndarray]], results_dir: str):
    metrics_figs = []
    display_labels = ["running", "cut"]
    plot_names = []
    all_true_values = []
    all_predictions = []
    for model, model_name, X_Test, Y_Test in models_and_data:
        predictions = model.predict(X_Test)
        true_values = Y_Test.flatten()

        predictions = np.apply_along_axis(lambda x: int(x > 0.50), 1, predictions)
        print(f"Predictions : {predictions}")
        print(f"True values : {true_values}")

        accuracy, uar, prec, fscore, recall_per_class, prec_per_class, fscore_per_class = calc_metrics(true_values,
                                                                                                       predictions)

        model.metadata["accuracy"] = accuracy
        model.metadata["recall_score"] = uar
        model.metadata["fscore"] = fscore
        model.metadata["recall_per_class"] = recall_per_class.tolist()
        model.metadata["prec_per_class"] = prec_per_class.tolist()
        model.metadata["fscore_per_class"] = fscore_per_class.tolist()
        model.metadata["true_values"] = true_values.tolist()
        model.metadata["predictions"] = predictions.tolist()

        model.save(os.path.join(results_dir, model_name))

        all_predictions.append(predictions)
        all_true_values.append(true_values)

        plot_names.append(model_name)

        metrics_fig, data_values = model.get_metrics_fig()
        metrics_figs.append(metrics_fig)

    all_true_values = np.asarray(all_true_values)
    confusion_fig, axs = create_confusion_matrices_from_values(true_values=all_true_values,
                                                               predicted_values=all_predictions,
                                                               plot_names=plot_names,
                                                               display_labels=display_labels,
                                                               normalize=None,
                                                               show=False
                                                               )
    metrics_figs.append(confusion_fig)

    create_pdf_from_figures(Path(results_dir, f"results_{datetime.datetime.now()}.pdf"),
                            metrics_figs,
                            plt_close_all=True)


def create_sequences(df, scaler, model_name, results_dir, window_len: int, overlap: int, train: bool):
    X_data, Y_data = df.loc[:, df.columns != 'Label'].values, df["Label"].values
    X_data = standardize_data(scaler,
                              X_data,
                              train=True,
                              model_name=model_name,
                              results_dir=results_dir)

    sequences, labels = cut_data_into_windows(X_data, Y_data, window_len, overlap, train)
    return sequences, labels


def run():
    dataset_samples = load_dataset("ACSS")
    dataset_dfs = [x.single_df_without_time for x in dataset_samples]

    results_dir = "evaluation"
    os.makedirs(results_dir, exist_ok=True)
    window_length = 64 * 2 # 4 seconds
    overlap = (window_length // 2) + 30
    features_num = 9
    input_shape = (window_length, features_num)
    epochs = 100
    models_and_data = []
    limit = 10
    count = 0
    for train_df, validation_df, test_df in next_cross_validation_split(dataset_dfs, limit=limit):
        model_name = f"LSTM_MODEL_{count}"

        scaler = StandardScaler()

        train_sequences, train_labels = create_sequences(train_df, scaler, model_name, results_dir,
                                                         window_length, overlap, True)

        validation_sequences, validation_labels = create_sequences(validation_df,
                                                                   scaler, model_name, results_dir,
                                                                   window_length, overlap, False)
        test_sequences, test_labels = create_sequences(test_df,
                                                       scaler, model_name, results_dir,
                                                       window_length, overlap, False)

        print("Train set : ")
        print_class_distribution(train_labels)
        print("Validation set : ")
        print_class_distribution(validation_labels)
        print("Test set : ")
        print_class_distribution(test_labels)

        model = LSTMModelWrapper(custom_lstm_model(input_shape,
                                                   model_name=None
                                                   ))

        model.metadata["overlap"] = overlap
        model.metadata["window_length"] = window_length
        model.metadata["input_shape"] = input_shape

        model.fit(train_sequences, train_labels, validation_sequences,
                  validation_labels,
                  epochs=epochs,
                  callbacks=default_callbacks(results_dir, model_name),
                  weighted=False
                  )
        predictions = model.predict(test_sequences)
        predictions = np.apply_along_axis(lambda x: int(x > 0.50), 1, predictions)

        evaluation_results = get_evaluation_results(model_name,
                                                    true_values=test_labels.flatten(),
                                                    predictions=predictions,
                                                    overlap=overlap,
                                                    epochs=epochs,
                                                    window_length=window_length,

                                                    )
        with open(Path(results_dir, f"{model_name}.json").path, "w+") as f:
            json.dump(evaluation_results, f, indent=4, sort_keys=True)

        count += 1
        # models_and_data.append(tuple([model, model_name, test_sequences, test_labels]))

    # create_pdfs(models_and_data, results_dir)


if __name__ == "__main__":
    run()
