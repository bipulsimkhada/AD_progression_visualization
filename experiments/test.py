from pathlib import Path
import json
import joblib
import numpy as np
import keras
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from alz_prog_net.eval import evaluate_model
from experiments.constants import MODALITIES
from utils import split_modalities
from visualizer.adpg import adpg_visualizer


def evaluate_ensemble(
    path,
    X_test,
    y_test,
    metadata_test,
    combo,
    run_name,
    pipeline_path,
):
    """
    Load all .keras models from a directory, preprocess the data using fold-specific
    pipelines, generate predictions from each model, and evaluate the ensemble.
    """
    model_path = Path(path)
    pipeline_path = Path(pipeline_path)

    # --------------------------------------------------------------
    # Predict with every model fold
    # --------------------------------------------------------------
    predictions = []

    for index in range(10):
        # 1. Load fold pipeline and transform raw test data
        pipe = joblib.load(pipeline_path / f"fold_{index}_pipeline.joblib")
        X_test_scaled = pipe.transform(X_test)

        # 2. Split preprocessed features into modality inputs
        X_test_modalities = split_modalities(
            X_test_scaled,
            combo,
        )

        # 3. Load fold model
        model = keras.models.load_model(
            model_path / f"model_{index}.keras",
            compile=False,
        )

        # 4. Generate predictions
        y_pred = model.predict(
            X_test_modalities,
            verbose=0,
        )
        predictions.append(y_pred["predictions"])

        # Free GPU/CPU memory across iterations
        keras.backend.clear_session()

    # --------------------------------------------------------------
    # Stack & Aggregate Predictions
    # Shape: (n_models, n_samples, n_outputs)
    # --------------------------------------------------------------
    y_pred_all = np.stack(predictions, axis=0)

    # Calculate ensemble mean trajectory and uncertainty (epistemic SD)
    y_pred_mean = np.mean(y_pred_all, axis=0)
    y_pred_sd = np.std(y_pred_all, axis=0)

    # --------------------------------------------------------------
    # Evaluate Ensemble
    # --------------------------------------------------------------
    y_test_stacked = np.stack(y_test[:, 0])
    evaluation = evaluate_model(
        y_test_stacked,
        y_pred_mean,
    )

    # Save metrics JSON
    results_dir = Path("results") / f"ensembled_{run_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / "result_test_set_ensemble.json"

    with open(result_path, "w") as f:
        json.dump(
            evaluation,
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Render Individual Trajectory Figures
    # --------------------------------------------------------------
    for index, (_, row) in enumerate(metadata_test.iterrows()):
        rid = row["RID"]
        viscode = row["VISCODE"]

        fig, ax = adpg_visualizer(
            y_test_stacked[index],
            y_pred_mean[index],
            y_pred_sd[index],
            rid,
            viscode,
        )

        fig_path = results_dir / f"figure_{index}_RID_{rid}_{viscode}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)