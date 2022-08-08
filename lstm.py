import io
import os

import keras.utils.np_utils
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import joblib
import json


class LSTMModelWrapper(object):
    def __init__(self,
                 custom_model: keras.Model
                 ):
        """
        Either a param dict or precompiled model can be provided
        """
        self.model = custom_model
        self.model.summary()
        self.__metadata = {}
        self.__model_histories = []
        self.__epochs = []
        self.__batch_sizes = []

    @property
    def name(self):
        return self.model.name

    @property
    def metadata(self):
        return self.__metadata

    @property
    def model_histories(self):
        return self.__model_histories

    @property
    def epochs(self):
        return self.__epochs

    @property
    def batch_sizes(self):
        return self.__batch_sizes

    def fit(self, X_train, y_train, X_val, y_val, batch_size: int = 10,
            epochs: int = 10, shuffle=False, weighted=False, callbacks=None):

        if weighted:
            class_weight = compute_class_weight("balanced",
                                                classes=np.unique(y_train),
                                                y=y_train)
            class_weight = {i: class_weight[i] for i in range(len(class_weight))}
        else:
            class_weight = None

        self.__batch_sizes.append(batch_size)
        self.__epochs.append(epochs)

        # callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model_history = self.model.fit(X_train, y_train,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       validation_data=(X_val, y_val),
                                       callbacks=callbacks,
                                       class_weight=class_weight,
                                       verbose=1,
                                       use_multiprocessing=True)

        self.__model_histories.append(model_history)

        self.__metadata["epochs"] = self.epochs
        self.__metadata["batch_sizes"] = self.batch_sizes
        self.__metadata["iterations"] = len(self.model_histories)

        return self

    def get_model_summary(self):
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path: str):
        joblib.dump(self, path, compress=True)
        name = os.path.split(path)[-1]
        name = f"{name}_metadata.json"
        base_dir = os.path.dirname(path)
        metadata_path = os.path.join(base_dir, name)
        data_values = self.get_metrics_values()
        for k in data_values.keys():
            self.metadata[f"model_history_{k}"] = data_values[k]
        with open(metadata_path, 'w+') as f:
            json.dump(self.metadata, f, indent=4, sort_keys=True)

    @classmethod
    def load(cls, name):
        return joblib.load(name)

    def get_metrics_values(self):
        data_values = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": []
        }
        losses = []
        val_losses = []
        accuracies = []
        val_accuracies = []
        # recalls = []
        # val_recalls = []
        for history in self.model_histories:
            h = history.history
            losses += h["loss"]
            val_losses += h["val_loss"]
            accuracies += h["accuracy"]
            val_accuracies += h["val_accuracy"]

        data_values["loss"] = losses
        data_values["val_loss"] = val_losses
        data_values["accuracy"] = accuracies
        data_values["val_accuracy"] = val_accuracies
        return data_values

    def get_metrics_fig(self):
        data_values = self.get_metrics_values()

        losses = data_values["loss"]
        val_losses = data_values["val_loss"]
        accuracies = data_values["accuracy"]
        val_accuracies = data_values["val_accuracy"]

        fig, (ax1, ax2) = plt.subplots(2, 1)
        # make a little extra space between the subplots
        fig.subplots_adjust(hspace=0.5)

        t = list(range(len(losses)))
        ax1.plot(t, losses)

        t = list(range(len(val_losses)))
        ax1.plot(t, val_losses)

        # ax1.set_xlim(0, 5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(['train', 'validation'], loc='upper left')
        ax1.grid(True)

        t = list(range(len(accuracies)))
        ax2.plot(t, accuracies)

        t = list(range(len(val_accuracies)))
        ax2.plot(t, val_accuracies)

        # ax2.set_xlim(0, 5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend(['train', 'validation'], loc='upper left')
        ax2.grid(True)
        fig.suptitle(f"Model : {self.model.name}", fontweight="bold", fontsize=24)
        fig.set_size_inches(20, 30)
        return fig, data_values
