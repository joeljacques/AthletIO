from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..utils import Path

from typing import List


def create_pdf_from_figures(output_path: Path, figures: List[Figure], plt_close_all=True) -> None:
    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(output_path.path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            if plt_close_all:
                plt.close('all')


__all__ = ["create_pdf_from_figures"]
