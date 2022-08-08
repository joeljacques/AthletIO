import os
from typing import List

import tqdm
import zipfile
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from resample import get_class_distribution_as_str


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


class ClassLabels(object):
    RUNNING = 0
    CUT = 1


class CutLabels(object):
    # Values are in milliseconds
    # ss:ms
    # Start the experiment : 00:35
    # Start running : 06:00
    # Cut : 09:67
    # Cut : 13:144
    # Cut : 16:917
    # Stop running : 20:391
    # Stop the recording 23:714
    start_time = 0
    start_running = (6 * 1000)
    first_cut = (9 * 1000) + 67
    second_cut = (13 * 1000) + 144
    third_cut = (16 * 1000) + 917
    stop_running = (20 * 1000) + 391
    stop_recording = (23 * 1000) + 714


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
        self.__number_of_samples = 0
        self.__duration = 0

    def __str__(self):
        return f"Dataset sample : {self.dir_path}"

    @property
    def number_of_samples(self):
        return self.__number_of_samples

    @property
    def duration(self):
        return round(self.__duration, 2)

    @property
    def frequency(self):
        return round(self.number_of_samples / self.duration, 2)

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

    def get_class_distribution(self):
        classes = self.single_df["Label"].values
        return get_class_distribution_as_str(classes)

    def print_info(self):
        print(self)
        print(f"Samples   : {self.number_of_samples}")
        print(f"Duration  : {self.duration} seconds")
        print(f"Frequency : {self.frequency} sample/second")
        print(self.get_class_distribution())
        print("-" * 30)

    def __create_single_df(self):
        to_drop = [self.accelerometer_df,
                   self.gyroscope_df,
                   self.linear_acceleration_df]
        to_drop_len = list(map(lambda x: len(x.values), to_drop))

        to_keep_idx = np.argmax(to_drop_len)
        dfs = []
        for idx, df in enumerate(to_drop):
            if idx == to_keep_idx:
                dfs.append(df)
            else:
                dfs.append(df.drop(['Time (s)'], axis=1))

        return pd.concat(dfs, axis=1)

    @classmethod
    def load(cls, dir_path: str, fixed_label: int):
        sample = cls()
        assert os.path.isdir(dir_path), f"The path : {dir_path} is not a directory"
        sample.__accelerometer_df = pd.read_csv(os.path.join(dir_path, "Accelerometer.csv"), sep=",")
        sample.__gyroscope_df = pd.read_csv(os.path.join(dir_path, "Gyroscope.csv"), sep=",")
        sample.__linear_acceleration_df = pd.read_csv(os.path.join(dir_path, "Linear Acceleration.csv"), sep=",")
        sample.__single_df = sample.__create_single_df()
        sample.__add_labels(fixed_label)

        sample.__meta_device_df = pd.read_csv(os.path.join(dir_path, "meta", "device.csv"), sep=",")
        sample.__meta_time_df = pd.read_csv(os.path.join(dir_path, "meta", "time.csv"), sep=",")
        sample.__dir_path = dir_path
        sample.__single_df.dropna(axis=0, inplace=True)

        sample.__single_df_without_time = sample.single_df.drop(['Time (s)'], axis=1)

        sample.__number_of_samples = len(sample.single_df.values)
        sample.__duration = sample.single_df["Time (s)"].max()
        sample.__duration = sample.__duration / 1000  # Convert to seconds
        # sample.print_info()
        return sample

    @staticmethod
    def __label_df(df, cuts: list, fixed_label: int):
        if fixed_label is not None:
            df["Label"] = fixed_label
            df["Time (s)"] *= 1000  # Convert to milliseconds
            return df
        df["Label"] = ClassLabels.RUNNING
        df["Time (s)"] *= 1000  # Convert to milliseconds

        for start, end in cuts:
            cut_df = df[(df['Time (s)'] >= start) & (df['Time (s)'] <= end)]
            assert len(cut_df.values) > 0, "Dataframe is empty"
            # print(f"Start : {start}, End: {end}, df size : {len(cut_df.values)}")

            df.loc[cut_df.index.values.tolist(), "Label"] = ClassLabels.CUT

        return df

    def __add_labels(self, fixed_label: int = None):
        cuts = [(CutLabels.start_running, CutLabels.first_cut),
                (CutLabels.first_cut, CutLabels.second_cut),
                (CutLabels.second_cut, CutLabels.third_cut),
                (CutLabels.third_cut, CutLabels.stop_running)
                ]
        self.__single_df = self.__label_df(self.single_df, cuts, fixed_label)


def load_dataset(dataset_path: str, fixed_label: int = None) -> List[DatasetSample]:
    samples_dirs = extract_dataset_dirs(dataset_path)
    dataset = []
    for sample_dir in samples_dirs:
        dataset.append(DatasetSample.load(sample_dir, fixed_label))
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
