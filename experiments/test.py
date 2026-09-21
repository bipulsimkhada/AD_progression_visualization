from pathlib import Path

import numpy as np
import keras
import json
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
    X_train,
    X_test,
    y_test,
    metadata_test,
    combo,
    run_name,
    imputer="mean",
    scaling="standard",
    
):
    """
    Load all .keras models from a directory, preprocess the data,
    generate predictions from each model, and evaluate the ensemble.

    Parameters
    ----------
    path : str or Path
        Directory containing the .keras models.

    X_train : array-like
        Full training feature matrix before preprocessing.

    y_train : array-like
        Training labels. Included for consistency/reference.

    X_test : array-like
        Full test feature matrix before preprocessing.

    y_test : array-like
        Test labels used for evaluation.

    combo : tuple
        Tuple describing the modalities used by the models.
        Passed to split_modalities().

    imputer : str
        SimpleImputer strategy, e.g. "mean", "median",
        "most_frequent", or "constant".

    scaling : str
        Either "standard" or "min-max".

    Returns
    -------
    dict
        Contains ensemble predictions, model-level predictions,
        prediction standard deviation, evaluation results, and
        preprocessing pipeline.
    """

    path = Path(path)

    # --------------------------------------------------------------
    # Find models
    # --------------------------------------------------------------

    model_paths = sorted(path.glob("*.keras"))

    if not model_paths:
        raise FileNotFoundError(
            f"No .keras models found in {path}"
        )

    print(f"Found {len(model_paths)} models.")

    # --------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------

    if scaling == "min-max":
        scaler = MinMaxScaler()
    elif scaling == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(
            "scaling must be either 'standard' or 'min-max'"
        )

    pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy=imputer),
            ),
            (
                "scaler",
                scaler,
            ),
        ]
    )

    # Fit ONLY on training data
    X_train_scaled = pipe.fit_transform(X_train)

    # Apply the exact same transformation to test data
    X_test_scaled = pipe.transform(X_test)

    # --------------------------------------------------------------
    # Split into modality inputs
    # --------------------------------------------------------------

    X_test_modalities = split_modalities(
        X_test_scaled,
        combo,
    )

    # --------------------------------------------------------------
    # Predict with every model
    # --------------------------------------------------------------

    predictions = []

    for model_path in model_paths:

        print(f"Predicting with {model_path.name}")

        model = keras.models.load_model(
            model_path,
            compile=False,
        )

        y_pred = model.predict(
            X_test_modalities,
            verbose=0,
        )

        predictions.append(y_pred)

    # --------------------------------------------------------------
    # Stack predictions
    #
    # Shape:
    #   (n_models, n_samples, n_outputs)
    #
    # or, depending on model output:
    #   (n_models, n_samples, ...)
    # --------------------------------------------------------------

    y_pred_all = np.stack(
        predictions,
        axis=0,
    )

    # --------------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------------

    y_pred_mean = np.mean(
        y_pred_all,
        axis=0,
    )

    y_pred_sd = np.std(
        y_pred_all,
        axis=0,
    )

    # --------------------------------------------------------------
    # Evaluate ensemble
    # --------------------------------------------------------------

    evaluation = evaluate_model(
        y_test,
        y_pred_mean,
    )

    results_dir = Path("results") / "cv" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"result_test_set_ensemble.json"
    
    with open(result_path, "w") as f:
        json.dump(
            evaluation,
            f,
            indent=2,
        )

    for index, (_, row) in enumerate(metadata_test.iterrows()):
        rid = row["RID"]
        viscode = row["VISCODE"]

        fig, ax = adpg_visualizer(
            y_test[index],
            y_pred_mean[index],
            y_pred_sd[index],
            rid,
            viscode,
        )

        fig_path = results_dir / f"figure_{index}_RID_{rid}_{viscode}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


