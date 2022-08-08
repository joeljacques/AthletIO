from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.utils.np_utils

import tensorflow as tf

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import io
import resample
from preprocessing import standardize_data
import os

def default_callbacks(result_dir: str, model_name: str):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(result_dir,f"model_{model_name}.h5"),
                        save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.06,
                          patience=4, min_lr=0.00001)
    ]
    return callbacks


def custom_lstm_model(input_shape, learning_rate=0.0001):
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
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"]
                  )
    return model


class LSTMModel(object):
    def __init__(self,
                 param_dict,
                 custom_model=None
                 ):
        """
        Either a param dict or precompiled model can be provided
        """
        if not custom_model:
            input_shape = param_dict["input_shape"]  # Tuple[int, int],
            num_classes = param_dict["num_classes"]
            metrics = param_dict["metrics"]
            model_name = param_dict["model_name"]
            learning_rate = param_dict["learning_rate"]

            self.model = keras.Sequential(
                [
                    layers.Input(input_shape),
                    layers.LSTM(256, return_sequences=False,
                                # kernel_regularizer=keras.regularizers.l2(0.001),
                                ),
                    layers.Dropout(0.3),
                    layers.Dense(1 if num_classes == 2 else num_classes,
                                 activation="sigmoid" if num_classes == 2 else "softmax"),
                ], name=model_name,
            )

            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy",
                               optimizer=optimizer,
                               metrics=metrics)
            self.model.summary()
        else:
            self.model = custom_model
            self.model.summary()

    def fit(self, X_train, y_train, X_val, y_val, batch_size: int = 10,
            epochs: int = 10, shuffle=False, weighted=False, callbacks=None):

        if weighted:
            class_weight = compute_class_weight("balanced",
                                                classes=np.unique(y_train),
                                                y=y_train)
            class_weight = {i: class_weight[i] for i in range(len(class_weight))}
        else:
            class_weight = None

        # callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.model_history = self.model.fit(X_train, y_train,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            validation_data=(X_val, y_val),
                                            callbacks=callbacks,
                                            class_weight=class_weight,
                                            verbose=1,
                                            use_multiprocessing=True)
        # self.model.save("first_lstm_model.h5")
        return self

    def get_model_summary(self):
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string

    def predict(self, x):
        return self.model.predict(x)

    # --------------------------------------------------------------
    # method for the model evaluation pipeline (evaluation.model_evaluation.ModelEvaluate)

    @staticmethod
    def execute_for_eval(X_train, y_train, X_test, y_test, subject, parameter_dict):
        num_classes = parameter_dict['num_classes']
        name = parameter_dict['name']
        custom_model_function = parameter_dict['custom_model_function']
        oversample_until_balanced = parameter_dict['oversample_until_balanced']
        undersample_until_balanced = parameter_dict['undersample_until_balanced']
        window_length = parameter_dict['window_length']
        result_path = parameter_dict['result_path']
        metrics = parameter_dict['metrics']
        learning_rate = parameter_dict['learning_rate']
        batch_size = parameter_dict['batch_size']

        print("....................")
        print(batch_size)

        epochs = parameter_dict['epochs']
        model_info = parameter_dict['model_info']
        num_folds = parameter_dict['num_folds']
        positive_threshold = parameter_dict['positive_threshold']

        # standardization
        # scaler = Standere
        # standardize_data(scaler,X_train,True,)
        X_train, X_test, scaler = preprocess.standardize_data(X_train, X_test)

        # resample the data
        if oversample_until_balanced:
            X_train, y_train = resample.oversample_sequences_until_balanced(X_train, y_train)
        elif undersample_until_balanced:
            X_train, y_train = resample.undersample_sequences_until_balanced(X_train, y_train)

        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=result_path + f"model_{subject[:4]}.h5",
                            save_best_only=True),

            ReduceLROnPlateau(monitor='val_loss', factor=0.06,
                              patience=4, min_lr=0.00001)
        ]
        """
        callbacks = None

        input_shape = (window_length, X_train.shape[2])

        param_dict = {"input_shape": input_shape,
                      "num_classes": num_classes,
                      "metrics": metrics,
                      "model_name": name,
                      "learning_rate": learning_rate}

        # model = custom_model_function(input_shape)
        model = None

        model_wrapper = LSTMModel(param_dict=param_dict, custom_model=model)
        model_wrapper.fit(X_train, y_train, X_test, y_test, batch_size, epochs, callbacks=callbacks)

        model = model_wrapper.model

        print("..............")
        print(X_train.shape, np.array(y_train).shape, X_test.shape, np.array(y_test).shape, batch_size, epochs)

        model.fit(X_train, y_train, X_test, y_test, batch_size, epochs)

        preds_test = model.predict(X_test)
        preds_test = (preds_test > positive_threshold).astype(np.int32).reshape(-1)

        preds_train = model.predict(X_train)
        preds_train = (preds_train > positive_threshold).astype(np.int32).reshape(-1)

        preds_proba = model.predict(X_test)

        """
    loss_learning_curve = plot.plot_loss_learning_curve(base_lstm.model_history, subject, epochs)
        accuracy_learning_curve = plot.plot_accuracy_learning_curve(base_lstm.model_history, subject,
                                                                    epochs)

        if not model_info.hyperparameter:
            model_info.hyperparameter = {
                "learning rate": learning_rate,
                "batch size": batch_size,
                "epochs": epochs,
                "units": units,
                "dropout": dropout,
            }

            model_info.vars["count"] = 0
            model_info.preprocessing["resample"] = "no resampling"
            model_info.architecture = model.get_model_summary()
            model_info.vars["histories"] = [(subject, model.model_history)]

        model_info.vars["count"] += 1
        model_info.vars["histories"].append((subject, model.model_history))

        if model_info.vars["count"] == num_folds-1:
            img1 = plot.lstm_acc_lc_subplots("LSTM Accuracy Learning Curve", model_info.vars["histories"])
            img2 = plot.lstm_loss_lc_subplots("LSTM Loss Learning Curve", model_info.vars["histories"])
            model_info.images.append(img1)
            model_info.images.append(img2)
        """

        return preds_test, preds_proba, preds_train, y_train
