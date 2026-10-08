from pathlib import Path
import json
import joblib
import numpy as np
import keras
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

from alz_prog_net.eval import evaluate_model
from utils import split_modalities
from visualizer.adpg import adpg_visualizer
from alz_prog_net.metrics import grouped_categorical_accuracy

def process_model(
    model_file,
    X_test_modalities,
    y_test,
    y_stable
):
    model = None
    output = None

    try:
        # ----------------------------------------------------------
        # 1. Load model
        # ----------------------------------------------------------
        model = keras.models.load_model(
            model_file,
            compile=True,
            custom_objects={
                "grouped_categorical_accuracy": [
                    grouped_categorical_accuracy
                ]
            },
        )

        # ----------------------------------------------------------
        # 2. Detailed prediction
        # ----------------------------------------------------------
        output = model(
            X_test_modalities,
            training=False,
            return_details=True,
        )

        y_pred = output["predictions"].numpy()

        # ----------------------------------------------------------
        # 3. Extract detailed outputs
        # ----------------------------------------------------------
        attention_weights = np.asarray(
            output["attention_weights"]
        )

        stable_mask = np.asarray(y_stable).astype(bool)

        attention_weights_stable = attention_weights[stable_mask].mean(axis=0)
        attention_weights_converter = attention_weights[~stable_mask].mean(axis=0)

        
        residual_scale = [
            np.asarray(level[0]).tolist() if level else None
            for level in output["residual_scales"]
        ]



        gate_stable = [
            time.numpy()[stable_mask].mean(axis=0).tolist()
            if time.numpy().size > 0 else []
            for level in output["gates"]
            for time in level
        ]

        gate_converter = [
            time.numpy()[~stable_mask].mean(axis=0).tolist()
            if time.numpy().size > 0 else []
            for level in output["gates"]
            for time in level
        ]

        delta_magnitude_stable = [
            time.numpy()[stable_mask].mean(axis=0).tolist()
            if time.numpy().size > 0 else []
            for level in output["delta_magnitudes"]
            for time in level
        ]

        delta_magnitude_converter = [
            time.numpy()[~stable_mask].mean(axis=0).tolist()
            if time.numpy().size > 0 else []
            for level in output["delta_magnitudes"]
            for time in level
        ]

        # ----------------------------------------------------------
        # 4. Targets
        # ----------------------------------------------------------
        y_true = np.stack(y_test[:, 0])

        loss_fn = model.loss["predictions"]

        if isinstance(loss_fn.class_weights, dict):
            class_weights = loss_fn.class_weights

            if class_weights.get("class_name"):
                class_weights = class_weights["config"]["value"]

            loss_fn.class_weights = class_weights

        subject_loss = loss_fn.call(
            y_true,
            y_pred,
        ).numpy()

        subject_loss_stable = subject_loss[stable_mask].mean(axis=0)
        subject_loss_converter = subject_loss[~stable_mask].mean(axis=0)

        # ----------------------------------------------------------
        # Return only what the caller needs
        # ----------------------------------------------------------
        return {
            "predictions": y_pred.tolist(),
            "attention_weights_stable": attention_weights_stable.tolist(),
            "attention_weights_converter": attention_weights_converter.tolist(),

            "residual_scale": residual_scale,

            "gate_stable": gate_stable,
            "gate_converter": gate_converter,

            "delta_magnitude_stable": delta_magnitude_stable,
            "delta_magnitude_converter": delta_magnitude_converter,

            "subject_loss": subject_loss.tolist(),
            "subject_loss_stable": subject_loss_stable.tolist(),
            "subject_loss_converter": subject_loss_converter.tolist(),
        }

    finally:
        # Remove TensorFlow/Keras objects before returning
        del output
        del model


def evaluate_ensemble(
    path,
    X_test,
    y_test,
    y_stable,
    metadata_test,
    combo,
    run_name,
    pipeline_path,
):
    """
    Load all .keras models from a directory, preprocess the data using
    fold-specific pipelines, generate detailed predictions from each model,
    evaluate the ensemble, and save per-model diagnostic information.

    Saved outputs:

        result_test_set_ensemble.json
            Ensemble evaluation metrics.

        model_details_test_set.json
            Per-model:
                - predictions
                - loss
                - stable loss
                - converter loss
                - attention weights
                - hidden states
                - residual scales
                - gates
                - delta magnitudes
                - subject metadata
    """

    model_path = Path(path)
    pipeline_path = Path(pipeline_path)

    # --------------------------------------------------------------
    # Results directory
    # --------------------------------------------------------------
    results_dir = Path("results") / f"ensembled_{run_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Containers
    # --------------------------------------------------------------
    predictions = []

    # --------------------------------------------------------------
    # Evaluate every fold model
    # --------------------------------------------------------------
    for index in range(10):

        print(f"Evaluating model {index}...")

        # ----------------------------------------------------------
        # 1. Load fold pipeline
        # ----------------------------------------------------------
        pipe = joblib.load(
            pipeline_path / f"fold_{index}_pipeline.joblib"
        )

        X_test_scaled = pipe.transform(X_test)

        # ----------------------------------------------------------
        # 2. Split into modality inputs
        # ----------------------------------------------------------
        X_test_modalities = split_modalities(
            X_test_scaled,
            combo,
        )

        # ----------------------------------------------------------
        # 3. Load model WITH compilation information
        # ----------------------------------------------------------
        model_file = model_path / f"model_{index}.keras"

        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                process_model,

                model_file,
                X_test_modalities,
                y_test,
                y_stable
            )

            result = future.result()

        predictions.append(np.asarray(result["predictions"]))

        # --------------------------------------------------------------
        # Save model details
        # --------------------------------------------------------------
        details_path = (
            results_dir
            / f"model_details_test_set_{index}.json"
        )

    
        with open(
            details_path,
            "w",
        ) as f:
    
            json.dump(
                result,
                f,
                indent=2,
                )


    # --------------------------------------------------------------
    # Stack predictions
    #
    # Shape:
    #     (n_models, n_samples, n_outputs)
    # --------------------------------------------------------------
    y_pred_all = np.stack(
        predictions,
        axis=0,
    )

    # --------------------------------------------------------------
    # Ensemble mean
    # --------------------------------------------------------------
    y_pred_mean = np.mean(
        y_pred_all,
        axis=0,
    )

    # --------------------------------------------------------------
    # Ensemble epistemic uncertainty
    # --------------------------------------------------------------
    y_pred_sd = np.std(
        y_pred_all,
        axis=0,
    )

    # --------------------------------------------------------------
    # Evaluate ensemble
    # --------------------------------------------------------------
    y_test_stacked = np.stack(
        y_test[:, 0]
    )

    evaluation = evaluate_model(
        y_test_stacked,
        y_pred_mean,
    )

    # --------------------------------------------------------------
    # Save ensemble metrics
    # --------------------------------------------------------------
    result_path = (
        results_dir
        / "result_test_set_ensemble.json"
    )

    with open(
        result_path,
        "w",
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Render individual trajectory figures
    # --------------------------------------------------------------
    for index, (_, row) in enumerate(
        metadata_test.iterrows()
    ):

        rid = row["RID"]
        viscode = row["VISCODE"]

        fig, ax = adpg_visualizer(
            y_test_stacked[index],
            y_pred_mean[index],
            y_pred_sd[index],
            rid,
            viscode,
        )

        fig_path = (
            results_dir
            / f"figure_{index}_RID_{rid}_{viscode}.png"
        )

        fig.savefig(
            fig_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    print()
    print("Evaluation complete.")
    print(
        f"Ensemble metrics: {result_path}"
    )
    print(
        f"Model details:    {details_path}"
    )

    return {
        "evaluation": evaluation,
        "ensemble_predictions": y_pred_mean,
        "ensemble_sd": y_pred_sd,
    }