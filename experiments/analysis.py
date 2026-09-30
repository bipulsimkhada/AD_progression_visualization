from pathlib import Path
import json
import pandas as pd
from experiments.constants import CONFIGS

# config_names = [
#     "proposed_with_mean_standarization",
#     "proposed_with_mean_standarization_base_loss",
#     "proposed_with_base_and_transition_loss_weight",
#     "proposed_with_base_and_converter_sample_weight",
#     "proposed_with_base_and_transition_weight"
# ]

# config_names = [
#     "proposed_with_mean_standarization",
#     "proposed_with_mean_standarization_wo_transformer",
#     "proposed_with_mean_standarization_wo_temporal",
#     "proposed_with_mean_standarization_wo_time",
#     "proposed_with_mean_standarization_base_model"
# ]

config_names = [
    "proposed_with_mean_standarization",
    "proposed_with_mean_standarization_wo_mri",
    "proposed_with_mean_standarization_wo_pet",
    "proposed_with_mean_standarization_wo_cog",
    "proposed_with_mean_standarization_wo_csf",
    "proposed_with_mean_standarization_wo_rf"
]



# config_names = [
#     "proposed_median_min-max",
#     "proposed_with_mean_min-max",
#     "proposed_with_mean_standarization",
#     "proposed_with_median_standarization"
# ]

def get_results(path, prefix=None):
    path = Path(path)

    rows = []

    # Find directories whose name starts with prefix
    if prefix:
        dirs = sorted(
            d for d in path.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        )
    else:
        dirs = sorted(
            d for d in path.iterdir()
            if d.is_dir() and d.name in config_names
        )

    if not dirs:
        raise FileNotFoundError(
            f"No directories starting with '{prefix}' found in {path}"
        )

    for result_dir in dirs:
        json_files = sorted(result_dir.glob("*.json"))

        if not json_files:
            continue

        results = []

        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)

            results.append({
                "overall_accuracy": data["all"]["overall"]["accuracy"],
                "overall_balanced_accuracy": data["all"]["overall"]["balanced_accuracy"],

                "stable_accuracy": data["stable"]["overall"]["accuracy"],
                "stable_balanced_accuracy": data["stable"]["overall"]["balanced_accuracy"],

                "stable_T0_accuracy": data["stable"]["by_time"]["current"]["accuracy"],
                "stable_T6_accuracy": data["stable"]["by_time"]["6_month"]["accuracy"],
                "stable_T12_accuracy": data["stable"]["by_time"]["12_month"]["accuracy"],
                "stable_T24_accuracy": data["stable"]["by_time"]["24_month"]["accuracy"],

                "converter_accuracy": data["converter"]["overall"]["accuracy"],
                "converter_balanced_accuracy": data["converter"]["overall"]["balanced_accuracy"],

                "converter_T0_accuracy": data["converter"]["by_time"]["current"]["accuracy"],
                "converter_T6_accuracy": data["converter"]["by_time"]["6_month"]["accuracy"],
                "converter_T12_accuracy": data["converter"]["by_time"]["12_month"]["accuracy"],
                "converter_T24_accuracy": data["converter"]["by_time"]["24_month"]["accuracy"],
            })

        df = pd.DataFrame(results)

        row = {
            "name": result_dir.name,
        }

        # Calculate mean ± SD
        for metric in df.columns:
            mean = df[metric].mean()
            sd = df[metric].std()

            row[f"{metric}"] = f"{mean:.4f} ± {sd:.4f}"

        # Numeric value used to identify best overall accuracy
        row["_overall_accuracy_mean"] = df["overall_accuracy"].mean()

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    # Best overall accuracy first
    summary = summary.sort_values(
        "_overall_accuracy_mean",
        ascending=False
    ).reset_index(drop=True)

    # Remove helper column
    summary = summary.drop(columns="_overall_accuracy_mean")
    return summary
