from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from ..utils import logger


def create_confusion_matrices_from_values(true_values: List[list] | np.ndarray | List[np.ndarray],
                                          predicted_values: List[list] | np.ndarray | List[np.ndarray],
                                          plot_names: List[str],
                                          display_labels: List[str] | None,
                                          normalize="true",  # 'true', 'pred', 'all', None
                                          cmap="viridis",
                                          main_title="Confusion Matrices",
                                          show=False) -> Tuple[Figure, Axes]:
    true_values = np.asarray(true_values)
    predicted_values = np.asarray(predicted_values)
    assert true_values.shape == predicted_values.shape
    count = true_values.shape[0]
    assert len(plot_names) == count
    if display_labels is not None:
        for idx, true_values_chunk in enumerate(true_values):
            if len(np.unique(true_values_chunk)) != len(display_labels):
                logger.warning(
                    f"True unique values in the plot [{plot_names[idx]}]"
                    f" are not equal to the number of classes in the display_labels array")

    if count % 2 == 0:
        rows_count = count // 2
    else:
        rows_count = (count + 1) // 2

    fig, axs = plt.subplots(rows_count, 1 if (rows_count == 1) and (count != 2) else 2)

    col_idx = 0
    row_idx = 0
    idx = 0

    while idx < count:
        if rows_count == 1:
            current_ax = axs[idx] if type(axs) is np.ndarray else axs
        else:
            current_ax = axs[row_idx, col_idx]
        unique_true_values = np.unique(true_values[idx])
        unique_predicated_values = np.unique(predicted_values[idx])
        if display_labels is not None:
            if len(unique_true_values) != len(display_labels):
                logger.warning(f"True unique values of the plot [{plot_names[idx]}] "
                               f"are not equal to the display labels array length {len(unique_true_values)}"
                               f" != {len(display_labels)}")
            if len(unique_predicated_values) != len(display_labels):
                logger.warning(f"Predicated unique values of the plot [{plot_names[idx]}] "
                               f"are not equal to the display labels array length {len(unique_predicated_values)}"
                               f" != {len(display_labels)}")

        ConfusionMatrixDisplay.from_predictions(true_values[idx],
                                                predicted_values[idx],
                                                display_labels=display_labels,
                                                normalize=normalize,
                                                ax=current_ax,
                                                cmap=cmap)
        current_ax.set_title(plot_names[idx],
                             fontweight="bold",
                             size=20)

        col_idx += 1
        if col_idx > 1:
            col_idx = 0
            row_idx += 1
        idx += 1

    if count > 2 and count % 2 != 0:
        # FIXME : Bug
        # axs[rows_count - 1, rows_count - 1].set_visible(False)
        pass
    fig.suptitle(main_title, fontweight="bold", fontsize=24)
    if show:
        plt.tight_layout()
        plt.show()
    else:
        fig.set_size_inches(20, 30)
        pass
    return fig, axs


__all__ = ["create_confusion_matrices_from_values"]
