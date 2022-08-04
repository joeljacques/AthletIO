import numpy as np
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def calc_metrics(y, preds, round_b=False, num_classes=2):
    accuracy = accuracy_score(y, preds)

    # we use unweighted average so that all classes get considered equally
    uar = recall_score(y, preds, labels=np.arange(num_classes), average='macro')
    prec = precision_score(y, preds, labels=np.arange(num_classes), average='macro')
    fscore = f1_score(y, preds, average="macro")

    # get scores for each class
    recall_per_class = recall_score(y, preds, labels=np.arange(num_classes), average=None)
    prec_per_class = precision_score(y, preds, labels=np.arange(num_classes), average=None)
    fscore_per_class = f1_score(y, preds, labels=np.arange(num_classes), average=None)

    if round_b:
        return round(accuracy, 2), round(uar, 2), round(prec, 2), round(fscore, 2), recall_per_class, prec_per_class, fscore_per_class
    else:
        return accuracy, uar, prec, fscore, recall_per_class, prec_per_class, fscore_per_class
