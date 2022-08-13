import json
from typing import List, Tuple

from scikit_pipelines import Path, create_confusion_matrices_from_values
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compare_models_plot(titel: str, results: np.ndarray, names: List[str]):
    print(np.asarray(results))
    print(np.asarray(names))
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle(titel)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)

    return fig


def plot_stat(stat_name: str, dfs: List[Tuple[str, pd.DataFrame]]):
    results = [df[stat_name].values for model_name, df in dfs]
    names = [model_name for model_name, df in dfs]
    return compare_models_plot(stat_name.capitalize(), np.asarray(results).T, names)


def plot_confusion_matrix(dfs: List[Tuple[str, pd.DataFrame]]):
    all_true_values = []
    all_predictions = []
    plot_names = []
    for model_name, df in dfs:
        idx = df["accuracy"].idxmax()
        true_values = df.iloc[idx]["true_values"]
        predictions = df.iloc[idx]["predictions"]
        all_true_values.append(true_values)
        all_predictions.append(predictions)
        plot_names.append(model_name)
    fig, axs = create_confusion_matrices_from_values(all_true_values,
                                                     all_predictions,
                                                     plot_names,
                                                     ["Running", "Cut"]
                                                     )
    return fig, axs


def run():
    evaluation_results = Path("evaluation")
    sgd_stats_files = []
    lstm_stats_files = []
    random_forest_stats_files = []

    def walk_callback(root, subdirs, files: List[str]):
        for f in files:
            if f.endswith(".json"):
                f = Path("evaluation", f).path
                if "sgd" in f.lower():
                    sgd_stats_files.append(f)
                elif "lstm" in f.lower():
                    lstm_stats_files.append(f)
                elif "random" in f.lower():
                    random_forest_stats_files.append(f)
                else:
                    raise Exception("WTF")

    evaluation_results.walk(walk_callback, True)

    sgd_stats = pd.DataFrame([json.load(open(x, "r")) for x in sgd_stats_files])
    lstm_stats = pd.DataFrame([json.load(open(x, "r")) for x in lstm_stats_files])
    random_forest_stats = pd.DataFrame([json.load(open(x, "r")) for x in random_forest_stats_files])
    models = [("SGD", sgd_stats),
              ("RandomForest", random_forest_stats),
              ("LSTM", lstm_stats)
              ]
    plot_confusion_matrix(models)

    metrics = ["accuracy", "fscore", "recall_score"]
    for metric in metrics:
        plot_stat(metric, models)

    plt.show()


if __name__ == "__main__":
    run()
