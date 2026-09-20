import multiprocessing
import os
from experiments.constants import LOSS_SEARCH_STAGES, RANDOM_STATE
from experiments.cv import cross_validation


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODALITIES = ("mri", "pet", "cog", "csf", "rf")

N_SPLITS = 10
EPOCHS = 250
BATCH_SIZE = 32

IMPUTER = "median"
SCALING = "min-max"

SEVERITY_MATRIX = [
    [0.0, 0.5, 2.0],
    [0.5, 0.0, 1.0],
    [2.0, 1.0, 0.0],
]


# -----------------------------------------------------------------------------
# Run one cross-validation experiment
# -----------------------------------------------------------------------------

def run_cv(
    X_train,
    y_train,
    y_stable_train,
    groups_train,
    config,
):
    """
    Run one cross-validation configuration.

    This function executes inside a separate process.
    """

    # os.environ["KERAS_BACKEND"] = "tensorflow"

    # import tensorflow as tf

    # # TensorFlow is imported only inside the child process.
    # devices = tf.config.list_physical_devices("GPU")

    # print("PID:", os.getpid())
    # print("Num GPUs Available:", len(devices))
    # print("GPUs:", devices)

    # for device in devices:
    #     tf.config.experimental.set_memory_growth(
    #         device,
    #         True,
    #     )

    # from experiments.cv import cross_validation


    config_name = config["name"]

    print("\n" + "=" * 80)
    print(f"STARTING: {config_name}")
    print("=" * 80)
    print("Config:", config)

    try:
        cross_validation(
            X_train,
            y_train,
            y_stable_train,
            groups_train,
            MODALITIES,
            run_name=f"loss/{config_name}",
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            imputer=IMPUTER,
            scaling=SCALING,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            time_weights=config["time_weights"],
            severity_matrix=SEVERITY_MATRIX,
            severity_weight=config["severity_weight"],
            transition_weight=config["transition_weight"],
            transition_loss=config["transition_loss"],
            huber_delta=config["huber_delta"],
            from_logits=False,
        )

        print("\n" + "=" * 80)
        print(f"FINISHED: {config_name}")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(f"FAILED: {config_name}")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 80)

        # Make sure the parent process knows this experiment failed.
        raise


# -----------------------------------------------------------------------------
# Run loss search stage
# -----------------------------------------------------------------------------

def run_loss_stage(
    stage_name,
    X_train,
    y_train,
    y_stable_train,
    groups_train,
):
    """
    Run every loss configuration in the stage.

    Each configuration is executed in its own fresh process.
    """

    configs = LOSS_SEARCH_STAGES[stage_name]

    print("\n" + "#" * 80)
    print(f"LOSS SEARCH STAGE: {stage_name}")
    print(f"Configurations: {len(configs)}")
    print("#" * 80)

    for i, config in enumerate(configs, start=1):
        config_name = config["name"]

        print("\n" + "-" * 80)
        print(
            f"Configuration {i}/{len(configs)}: "
            f"{config_name}"
        )
        print("-" * 80)

        # multiprocessing.set_start_method("spawn", force=True)
        process = multiprocessing.Process(
            target=run_cv,
            args=(
                X_train,
                y_train,
                y_stable_train,
                groups_train,
                config,
            ),
            name=f"loss-{config_name}",
        )

        process.start()

        print(
            f"Started process PID={process.pid} "
            f"for {config_name}"
        )

        # Wait until this experiment completely finishes
        # before starting the next one.
        process.join()

        if process.exitcode == 0:
            print(
                f"\nSUCCESS: {config_name} "
                f"(PID={process.pid})"
            )
        else:
            print(
                f"\nFAILED: {config_name} "
                f"(PID={process.pid}, "
                f"exit code={process.exitcode})"
            )

            # Stop the entire loss search if one experiment fails.
            raise RuntimeError(
                f"Loss configuration '{config_name}' failed "
                f"with exit code {process.exitcode}"
            )
