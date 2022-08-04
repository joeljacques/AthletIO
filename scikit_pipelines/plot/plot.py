import numpy as np


def print_class_distribution(y):
    print(get_class_distribution_as_str(y))


def get_class_distribution_as_str(y):
    vals, counts = np.unique(y, return_counts=True)
    output = "Class distribution: "
    output += " " + str(vals)
    output += " " + str(counts)
    return output
