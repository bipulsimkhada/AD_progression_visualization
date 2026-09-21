import numpy as np
from experiments.constants import MODALITIES

def compute_time_class_weights(y_train):
    labels = np.argmax(y_train, axis=-1)

    num_times = labels.shape[1]
    num_classes = y_train.shape[-1]

    weights = np.zeros(
        (num_times, num_classes),
        dtype=np.float32
    )

    for t in range(num_times):
        counts = np.bincount(
            labels[:, t],
            minlength=num_classes
        )

        total = counts.sum()

        weights[t] = np.divide(
            total,
            num_classes * counts,
            out=np.zeros(num_classes, dtype=np.float32),
            where=counts != 0
        )

    return weights

def split_modalities(X, combo):
    return [
        X[:, MODALITIES[m][1]]
        for m in combo
        if m in MODALITIES
    ]