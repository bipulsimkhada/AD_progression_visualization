import pandas as pd
import numpy as np
from pathlib import Path

regions = ['Ventricles/ICV', 'Hippocampus/ICV', 'WholeBrain/ICV', 'Entorhinal/ICV', 'Fusiform/ICV', 'MidTemp/ICV',
           'FDG', 'AV45',
           'CDRSB', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting','RAVLT_perc_forgetting','FAQ', 'MOCA', 'LDELTOTAL', 'DIGITSCOR','TRABSCOR',
           'ABETA', 'PTAU', 'TAU', 'AGE', 'PTEDUCAT', 'APOE4', 'PTGENDER']
targets = ['DX', 'DX6', 'DX12', 'DX24']

DATASET_DIR = Path(__file__).resolve().parent

def create_dataset():
    file_path = DATASET_DIR / "dataset_with_outlier_removed.csv"
    df = pd.read_csv(file_path, low_memory=False, index_col="ID")
    df['PTGENDER'] = df['PTGENDER'].map({
        'Male': 0,
        'Female': 1
    })

    X = df[regions]
    y = df[targets].copy()
    groups = df['RID'].to_numpy()
    metadata = df[['RID', 'VISCODE']]

    y["stable"] = (y.nunique(axis=1) == 1).astype(int)

    return X, y, groups, metadata

def createOutputLabels(Labels):
    Labels = Labels[targets]

    n_samples = len(Labels)
    class_map = {
        "CN": 0,
        "MCI": 1,
        "AD": 2,
    }

    output = np.empty((n_samples, 3), dtype=object)

    for i in range(n_samples):
        states = np.zeros(4, dtype=np.int32)
        prediction = np.zeros((4,3), dtype=np.float32)

        for j in range(4):
            label = Labels.iloc[i,j]
            class_index = class_map[label]
            states[j] = class_index

            prediction[j, class_index] = 1.0

        #transitions
        delta = np.diff(states)
        has_forward = np.any(delta > 0)
        has_reverse = np.any(delta < 0)

        trajectory = np.zeros(4, dtype=np.float32)

        if (not has_forward and not has_reverse):
            trajectory_class = 0
        elif has_forward and not has_reverse:
            trajectory_class = 1
        elif not has_forward and has_reverse:
            trajectory_class = 2
        else:
            trajectory_class = 3

        trajectory[trajectory_class] = 1.0

        # conversion
        conversion = np.zeros((3, 2), dtype=np.float32)
        forward_indices = np.where(delta > 0)[0]

        if len(forward_indices) == 0:
            conversion[:, 0] = 0.0
            conversion[:, 1] = 1.0
        else:
            first_conversion = forward_indices[0]

            conversion[:first_conversion + 1, 1] = 1.0,
            conversion[first_conversion, 0] = 1.0

        output[i, 0] = prediction
        output[i, 1] = trajectory
        output[i, 2] = conversion

    return output
