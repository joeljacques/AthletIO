from utils import load_dataset, plot_df, plt
from preprocessing import next_cross_validation_split, standardize_data
from sklearn.preprocessing import StandardScaler
import os
import datetime
from sklearn.svm import OneClassSVM

def run():
    dataset_samples = load_dataset("ACSS")
    dataset_dfs = [x.single_df_without_time for x in dataset_samples]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for train_df, test_df, validation_df in next_cross_validation_split(dataset_dfs, limit=1):
        scaler = StandardScaler()
        model_name = str(datetime.datetime.now())
        X_train = standardize_data(scaler,
                                   train_df.values,
                                   train=True,
                                   model_name=model_name,
                                   results_dir=results_dir)

        X_test = standardize_data(scaler,
                                  test_df.values,
                                  train=False,
                                  model_name=model_name,
                                  results_dir=results_dir)

        X_validation = standardize_data(scaler,
                                        train_df.values,
                                        train=False,
                                        model_name=model_name,
                                        results_dir=results_dir)


if __name__ == "__main__":
    run()
