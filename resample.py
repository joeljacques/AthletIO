import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def get_class_distribution_as_str(y):
    vals, counts = np.unique(y, return_counts=True)
    output = "Class distribution: "
    output += " " + str(vals)
    output += " " + str(counts)
    return output


def print_class_distribution(y):
    print(get_class_distribution_as_str(y))


# oversamples sequences, that is 3d time series data of shape (num_samples, timesteps, num_features)
# sampling_strategy = 'auto' (all classes that are not the majority class get oversampled randomly until
# the dataset is balanced)
def oversample_sequences_until_balanced(X_train, y_train):
    print("Class distribution before oversampling")
    print_class_distribution(y_train)
    randomOverSampler = RandomOverSampler()
    randomOverSampler.fit_resample(X_train[:, :, 0], y_train)
    X_train = X_train[randomOverSampler.sample_indices_]
    y_train = y_train[randomOverSampler.sample_indices_]
    print("Class distribution after oversampling")
    print_class_distribution(y_train)
    return X_train, y_train


def undersample_sequences_until_balanced(X_train, y_train):
    print("Class distribution before undersampling")
    print_class_distribution(y_train)
    randomUnderSampler = RandomUnderSampler()
    randomUnderSampler.fit_resample(X_train[:, :, 0], y_train)
    X_train = X_train[randomUnderSampler.sample_indices_]
    y_train = y_train[randomUnderSampler.sample_indices_]
    print("Class distribution after undersampling")
    print_class_distribution(y_train)
    return X_train, y_train
