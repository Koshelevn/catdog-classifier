import numpy as np
from sklearn.metrics import f1_score


def accuracy(scores, labels, threshold=0.5):
    predicted = np.array(scores > threshold).astype(np.int32)
    return np.mean(predicted == labels)


def f1(scores, labels, threshold=0.5):
    predicted = np.array(scores > threshold).astype(np.int32)
    return f1_score(labels, predicted)


def calculate_metrics(scores, labels, print_log=False):
    """Compute all the metrics from tracked_metrics dict using scores and labels."""

    scores_array = np.array(scores).astype(np.float32)
    labels_array = np.array(labels)
    tracked_metrics = {"accuracy": accuracy, "f1-score": f1}

    metric_results = {}
    for k, v in tracked_metrics.items():
        metric_value = v(scores_array, labels_array)
        metric_results[k] = metric_value

    if print_log:
        print(" | ".join(["{}: {:.4f}".format(k, v) for k, v in metric_results.items()]))

    return metric_results
