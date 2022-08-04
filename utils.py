import os
from typing import List

import tqdm
import zipfile
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


def get_sorted_paths_in_dir(input_dir: str):
    assert os.path.exists(input_dir)
    assert os.path.isdir(input_dir)
    paths = os.listdir(input_dir)
    paths = [os.path.join(input_dir, p) for p in paths]
    paths = sorted(paths)
    return paths


def extract_dataset_dirs(dataset_path: str):
    paths = get_sorted_paths_in_dir(dataset_path)
    paths = [p for p in paths if os.path.isfile(p)]
    extracted_dirs = []
    for zip_file_path in tqdm.tqdm(paths, f"Extracting zip files in \"{dataset_path}\""):
        basename = os.path.basename(zip_file_path).split(".")[0]
        basedir = os.path.dirname(zip_file_path)
        dir_out = os.path.join(basedir, basename)
        extracted_dirs.append(dir_out)
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(dir_out)
        except Exception as e:
            print(f"[Error] Failed to extract files from {zip_file_path}. {e}")
            last_idx = len(extracted_dirs) - 1
            extracted_dirs.pop(last_idx)
    return extracted_dirs


class DatasetSample(object):
    def __init__(self):
        self.__accelerometer_df = None
        self.__gyroscope_df = None
        self.__linear_acceleration_df = None
        self.__meta_device_df = None
        self.__meta_time_df = None
        self.__dir_path = None
        self.__single_df = None
        self.__single_df_without_time = None

    def __str__(self):
        return f"Dataset sample : {os.path.basename(self.dir_path)}"

    @property
    def dir_path(self) -> str:
        return self.__dir_path

    @property
    def single_df(self) -> pd.DataFrame:
        return self.__single_df

    @property
    def single_df_without_time(self) -> pd.DataFrame:
        return self.__single_df_without_time

    @property
    def accelerometer_df(self) -> pd.DataFrame:
        return self.__accelerometer_df

    @property
    def gyroscope_df(self) -> pd.DataFrame:
        return self.__gyroscope_df

    @property
    def linear_acceleration_df(self) -> pd.DataFrame:
        return self.__linear_acceleration_df

    @property
    def meta_device_df(self) -> pd.DataFrame:
        return self.__meta_device_df

    @property
    def meta_time_df(self) -> pd.DataFrame:
        return self.__meta_time_df

    @classmethod
    def load(cls, dir_path: str):
        sample = cls()
        assert os.path.isdir(dir_path), f"The path : {dir_path} is not a directory"
        sample.__accelerometer_df = pd.read_csv(os.path.join(dir_path, "Accelerometer.csv"), sep=",")
        sample.__gyroscope_df = pd.read_csv(os.path.join(dir_path, "Gyroscope.csv"), sep=",")
        sample.__linear_acceleration_df = pd.read_csv(os.path.join(dir_path, "Linear Acceleration.csv"), sep=",")

        sample.__meta_device_df = pd.read_csv(os.path.join(dir_path, "meta", "device.csv"), sep=",")
        sample.__meta_time_df = pd.read_csv(os.path.join(dir_path, "meta", "time.csv"), sep=",")
        sample.__dir_path = dir_path
        sample.__single_df = pd.concat([sample.accelerometer_df,
                                        sample.gyroscope_df,
                                        sample.linear_acceleration_df],
                                       axis=1)
        sample.__single_df_without_time = pd.concat([sample.accelerometer_df.drop('Time (s)', axis=1),
                                                     sample.gyroscope_df.drop('Time (s)', axis=1),
                                                     sample.linear_acceleration_df.drop('Time (s)', axis=1)],
                                                    axis=1)

        return sample


def load_dataset(dataset_path: str) -> List[DatasetSample]:
    samples_dirs = extract_dataset_dirs(dataset_path)
    dataset = []
    for sample_dir in samples_dirs:
        dataset.append(DatasetSample.load(sample_dir))
    return dataset


def plot_df(df: pd.DataFrame, title: str):
    columns = df.columns.values.copy()
    values = []

    for col_idx in range(len(columns)):
        values.append(df[columns[col_idx]].values.tolist())
    values = np.asarray(values)
    time_steps = values[0]

    fig, plots = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    for col_idx in range(1, values.shape[0]):
        xs = time_steps
        ys = values[col_idx]
        label = columns[col_idx]
        plot = plots[col_idx - 1]
        plot.plot(xs, ys)
        plot.set_title(label)
    fig.tight_layout()
