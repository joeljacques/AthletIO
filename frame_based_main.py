from __future__ import annotations

import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_features
from preprocessing import next_cross_validation_split, \
    cut_data_into_windows
from preprocessing import standardize_data
from scikit_pipelines import Path
from utils import get_evaluation_results
from utils import load_dataset
from scikit_pipelines.plot import print_class_distribution
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def create_sequences(df,
                     window_len: int,
                     overlap: int, train: bool):
    # X_data are the features, Y_data are the labels
    X_data, Y_data = df.loc[:, df.columns != 'Label'].values, df["Label"]
    sequences, labels = cut_data_into_windows(X_data, Y_data, window_len, overlap, train)
    sequences = extract_features(sequences)
    return sequences, labels


def run():
    dataset_samples = load_dataset("ACSS")
    dataset_dfs = [x.single_df_without_time for x in dataset_samples]

    results_dir = "evaluation"
    os.makedirs(results_dir, exist_ok=True)
    window_length = 64 * 2
    overlap = (window_length // 2) + 30

    features_num = 9
    input_shape = (window_length, features_num)
    epochs = 1000
    models_and_data = []

    val_accuracies = []
    test_accuracies = []
    limit = 10
    count = 0
    loss_tol = 1e-2
    max_depth = 2
    random_state = 0
    for train_df, _, test_df in next_cross_validation_split(dataset_dfs, limit=limit):
        sgd_model_name = f"SGD_MODEL_{count}"
        random_forest_model_name = f"RANDOM_FOREST_MODEL_{count}"
        svm_model_name = F"LINEAR_SVC_MODEL_{count}"
        ada_model_name = F"ADA_MODEL_{count}"
        boosting_model_name = F"BOOSTING_MODEL_{count}"

        scaler = StandardScaler()

        train_sequences, train_labels = create_sequences(train_df,
                                                         window_length, overlap,
                                                         True)
        print("Train-set class-distribution : ")
        print_class_distribution(train_labels)

        train_sequences = standardize_data(scaler, train_sequences,
                                           train=True,
                                           model_name=f"SGD_AND_RANDOM_FOREST_{count}",
                                           results_dir=results_dir)
        test_sequences, test_labels = create_sequences(test_df,
                                                       window_length,
                                                       overlap, False)
        print("Test-set class-distribution : ")
        print_class_distribution(test_labels)

        test_sequences = standardize_data(scaler, test_sequences,
                                          train=False,
                                          model_name=f"SGD_AND_RANDOM_FOREST_{count}",
                                          results_dir=results_dir)

        train_labels = train_labels.flatten()

        test_labels = test_labels.flatten()

        # clf = svm.SVC(kernel="linear")
        pipelines = [
            (sgd_model_name, make_pipeline(scaler, SGDClassifier(max_iter=epochs,
                                                                 tol=loss_tol,
                                                                 verbose=1
                                                                 ))),
            (random_forest_model_name, make_pipeline(scaler,
                                                     RandomForestClassifier(max_depth=max_depth,
                                                                            random_state=random_state,
                                                                            verbose=1))),
            (svm_model_name, make_pipeline(
                scaler, LinearSVC(class_weight='balanced')
            )),
            (ada_model_name, make_pipeline(AdaBoostClassifier())),
            (boosting_model_name, make_pipeline(GradientBoostingClassifier()))
        ]
        for model_name, clf in pipelines:
            clf.fit(train_sequences, train_labels)
            test_pred = clf.predict(test_sequences)
            evaluation_results = get_evaluation_results(model_name,
                                                        true_values=test_labels,
                                                        predictions=test_pred,
                                                        loss_tol=loss_tol,
                                                        overlap=overlap,
                                                        epochs=epochs,
                                                        window_length=window_length,
                                                        max_depth=max_depth,
                                                        random_state=random_state
                                                        )
            with open(Path(results_dir, f"{model_name}.json").path, "w+") as f:
                json.dump(evaluation_results, f, indent=4, sort_keys=True)

        count += 1


if __name__ == "__main__":
    run()
