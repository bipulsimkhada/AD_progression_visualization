import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def adpg_visualizer(y_true, y_pred_mean, y_pred_sd, rid, viscode):
    """
    Visualize AD progression predictions for one sample.

    Rows:
        CN  -> green
        MCI -> blue
        D   -> red

    Color intensity is proportional to the predicted value.

    Parameters
    ----------
    y_true : array-like, shape (4, 3)
        Ground-truth values for:
        T0, T6, T12, T24 x CN, MCI, D

    y_pred_mean : array-like, shape (4, 3)
        Predicted mean probabilities/values.

    y_pred_sd : array-like, shape (4, 3)
        Predicted standard deviations.

    rid : int or float
        Subject RID.

    viscode : int, float, or str
        VISCODE associated with the sample.

    Returns
    -------
    fig, ax
    """

    # ------------------------------------------------------------
    # Convert to numpy arrays
    # ------------------------------------------------------------
    y_true = np.asarray(y_true)
    y_pred_mean = np.asarray(y_pred_mean)
    y_pred_sd = np.asarray(y_pred_sd)

    # ------------------------------------------------------------
    # Validate shapes
    # ------------------------------------------------------------
    expected_shape = (4, 3)

    for name, arr in [
        ("y_true", y_true),
        ("y_pred_mean", y_pred_mean),
        ("y_pred_sd", y_pred_sd),
    ]:
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {arr.shape}"
            )

    # ------------------------------------------------------------
    # Convert:
    #
    # (time, class) -> (class, time)
    # ------------------------------------------------------------
    mean_plot = y_pred_mean.T
    sd_plot = y_pred_sd.T

    # ------------------------------------------------------------
    # Class-specific colormaps
    #
    # Low value  -> very light color
    # High value -> saturated/dark color
    # ------------------------------------------------------------
    cn_cmap = LinearSegmentedColormap.from_list(
        "CN_green",
        ["#F1F8F2", "#A5D6A7", "#2E7D32"]
    )

    mci_cmap = LinearSegmentedColormap.from_list(
        "MCI_blue",
        ["#EFF6FF", "#90CAF9", "#1565C0"]
    )

    d_cmap = LinearSegmentedColormap.from_list(
        "D_red",
        ["#FFF1F1", "#EF9A9A", "#C62828"]
    )

    cmaps = [
        cn_cmap,
        mci_cmap,
        d_cmap
    ]

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7, 5)
    )

    # ------------------------------------------------------------
    # Draw each class row separately
    #
    # This allows every row to have its own color.
    # ------------------------------------------------------------
    for class_idx in range(3):

        row = mean_plot[class_idx]

        # Reshape into (1, time)
        row_data = row[np.newaxis, :]

        ax.imshow(
            row_data,
            aspect='auto',
            cmap=cmaps[class_idx],
            vmin=0,
            vmax=1,
            extent=[
                -0.5,
                3.5,
                class_idx + 0.5,
                class_idx - 0.5
            ],
            interpolation='nearest'
        )

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------
    try:
        rid_text = str(int(rid))
    except (ValueError, TypeError):
        rid_text = str(rid)

    try:
        viscode_text = f"m{int(viscode)}"
    except (ValueError, TypeError):
        viscode_text = str(viscode)

    ax.set_title(
        f"RID: {rid_text}, VISCODE: {viscode_text}",
        fontsize=12
    )

    # ------------------------------------------------------------
    # X-axis
    # ------------------------------------------------------------
    x_ticks = [0, 1, 2, 3]
    x_labels = ["T0", "T6", "T12", "T24"]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # ------------------------------------------------------------
    # Y-axis
    # ------------------------------------------------------------
    y_ticks = [0, 1, 2]
    y_labels = ["CN", "MCI", "D"]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    ax.tick_params(
        axis='both',
        which='both',
        length=0
    )

    # ------------------------------------------------------------
    # Show prediction mean values
    # ------------------------------------------------------------
    for class_idx in range(3):

        for time_idx in range(4):

            value = mean_plot[class_idx, time_idx]

            if np.isfinite(value):

                # Choose text color based on intensity
                text_color = "black"

                ax.text(
                    time_idx,
                    class_idx,
                    f"{value:.2f}",
                    color=text_color,
                    fontsize=13,
                    fontweight='semibold',
                    ha='center',
                    va='center'
                )

    # ------------------------------------------------------------
    # Vertical grid lines
    # ------------------------------------------------------------
    for x in range(1, len(x_ticks)):
        ax.axvline(
            x=x - 0.5,
            color='white',
            linewidth=1.2,
            zorder=5
        )

    # ------------------------------------------------------------
    # Horizontal grid lines
    # ------------------------------------------------------------
    for y in [0.5, 1.5]:
        ax.axhline(
            y=y,
            color='white',
            linewidth=1.2,
            zorder=5
        )

    # ------------------------------------------------------------
    # Outer border
    # ------------------------------------------------------------
    ax.axhline(
        y=-0.5,
        color='black',
        linewidth=0.8,
        zorder=6
    )

    ax.axhline(
        y=2.5,
        color='black',
        linewidth=0.8,
        zorder=6
    )

    # ------------------------------------------------------------
    # Ground-truth diagnosis
    # ------------------------------------------------------------
    true_classes = np.argmax(y_true, axis=1)

    true_labels = [
        y_labels[class_idx]
        for class_idx in true_classes
    ]

    table = ax.table(
        cellText=[true_labels],
        rowLabels=["Actual"],
        cellLoc='center',
        rowLoc='center',
        bbox=[0.0, -0.18, 1.0, 0.10]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():

        cell.set_edgecolor('black')
        cell.set_linewidth(0.6)

        if row == 0:
            cell.set_facecolor('#F2F2F2')
            cell.set_text_props(weight='bold')

    # ------------------------------------------------------------
    # Uncertainty strip
    # ------------------------------------------------------------
    ax.axhspan(
        -1,
        -0.5,
        facecolor='#FFD54F',
        zorder=0
    )

    ax.text(
        -0.63,
        -0.75,
        "±",
        color='black',
        fontsize=11,
        fontweight='bold',
        ha='center',
        va='center'
    )

    # Maximum SD across classes at each timepoint
    max_sd = np.max(sd_plot, axis=0)

    for time_idx in range(4):

        if np.isfinite(max_sd[time_idx]):

            ax.text(
                time_idx,
                -0.75,
                f"{max_sd[time_idx]:.2f}",
                color='black',
                fontsize=13,
                fontweight='semibold',
                ha='center',
                va='center'
            )

    # ------------------------------------------------------------
    # Plot limits
    # ------------------------------------------------------------
    ax.set_ylim(2.5, -1)

    # ------------------------------------------------------------
    # Spine formatting
    # ------------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_edgecolor('silver')
        spine.set_linewidth(1)

    plt.tight_layout()

    return fig, ax