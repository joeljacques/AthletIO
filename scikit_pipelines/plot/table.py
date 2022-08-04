import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple


def create_table_from_dict(input_dict: dict, show=False) -> Tuple[Figure, Axes]:
    out = {}
    for key in input_dict.keys():
        out[key] = [str(input_dict[key])]
    df = pd.DataFrame.from_dict(out)
    fig = plt.figure()
    axs = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    table_data = []
    for idx, col in enumerate(df.columns):
        table_data.append([col, df.values[0][idx]])

    table = axs.table(cellText=table_data,
                      # cellText=df.values,
                      #             colLabels=df.columns,
                      loc='center',
                      )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    fig.set_size_inches(10, 15)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, axs


__all__ = ["create_table_from_dict"]
