import multiprocessing
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from dataset.dataset import create_dataset, createOutputLabels
from experiments.constants import RANDOM_STATE, LOSS_SEARCH_STAGES, CONFIGS
from experiments.cv import cross_validation, save_preprocessing_pipeline
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
        n_splits=10,
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

    # for i, config in enumerate(CONFIGS, start=1):
    #     config_name = config["name"]

    #     print("\n" + "-" * 80)
    #     print(
    #         f"Configuration {i}/{len(CONFIGS)}: "
    #         f"{config_name}"
    #     )
    #     print("-" * 80)

    #     # multiprocessing.set_start_method("spawn", force=True)
    #     process = multiprocessing.Process(
    #         target=cross_validation,
    #         args=(
    #             X_train,
    #             y_train,
    #             y_stable_train,
    #             groups_train,
    #             config,
    #         ),
    #         name=f"loss-{config_name}",
    #     )

    #     process.start()

    #     print(
    #         f"Started process PID={process.pid} "
    #         f"for {config_name}"
    #     )

    #     # Wait until this experiment completely finishes
    #     # before starting the next one.
    #     process.join()

    #     if process.exitcode == 0:
    #         print(
    #             f"\nSUCCESS: {config_name} "
    #             f"(PID={process.pid})"
    #         )
    #     else:
    #         print(
    #             f"\nFAILED: {config_name} "
    #             f"(PID={process.pid}, "
    #             f"exit code={process.exitcode})"
    #         )

    #         # Stop the entire loss search if one experiment fails.
    #         raise RuntimeError(
    #             f"Loss configuration '{config_name}' failed "
    #             f"with exit code {process.exitcode}"
    #         )

    evaluate_ensemble(
        "models/proposed_with_mean_standarization",
        X_test, y_test, metadata_test,
        ("mri", "pet", "cog", "csf", "rf"),
        "ensemble_proposed_with_mean_standarization",
        "results/cv/mean_standarization_pipeline"
    )


    # save_preprocessing_pipeline(
    #     X_train,
    #     y_train,
    #     y_stable_train,
    #     groups_train,
    #     imputer="mean",
    #     scaling="standard"
    # )


if __name__ == "__main__":
    # Set the spawn context ONCE in the parent process before spawning children
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    
    main()