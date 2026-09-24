import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from dataset.dataset import create_dataset, createOutputLabels
from experiments.constants import RANDOM_STATE
from experiments.loss import run_loss_stage
from experiments.test import evaluate_ensemble

def main():
    os.environ["KERAS_BACKEND"] = "tensorflow"

    devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available:", len(devices))
    print(devices)

    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    print("=" * 80)
    print("CREATING DATASET")
    print("=" * 80)

    X, y, groups, metadata = create_dataset()

    target_tensors = createOutputLabels(y)
    y_stable = y["stable"].to_numpy()

    # -------------------------------------------------------------------------
    # 80% train / 20% test
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CREATING TRAIN / TEST SPLIT")
    print("=" * 80)

    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        sgkf.split(
            X,
            y_stable,
            groups,
        )
    )

    X_train = X.iloc[train_idx]
    y_stable_train = y_stable[train_idx]
    y_train = target_tensors[train_idx]
    groups_train = groups[train_idx]

    X_test = X.iloc[test_idx]
    y_test = target_tensors[test_idx]
    metadata_test = metadata.iloc[test_idx]

    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples:  {len(test_idx)}")

    # run_loss_stage("stage_3_time_weights", X_train, y_train, y_stable_train, groups_train)
    run_loss_stage("stage_4_loss_type", X_train, y_train, y_stable_train, groups_train)

    # evaluate_ensemble("models/loss/s4_loss_huber", X_train, X_test, y_test, metadata_test,
    #                         ("mri", "pet", "cog", "csf", "rf"),
    #                         "ensemble_test_s4_loss_huber",
    #                         "median", "min-max")


if __name__ == "__main__":
    # Set the spawn context ONCE in the parent process before spawning children
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    
    main()