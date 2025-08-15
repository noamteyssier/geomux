import pandas as pd
import numpy as np


def read_table(filename: str, sep: str = "\t") -> pd.DataFrame:
    """
    Reads an input file and confirms that
    the file is in an expected format
    """
    frame = pd.read_csv(
        filename,
        header=None,
        names=["barcode", "guide", "n_umi"],
        dtype={"barcode": str, "guide": str, "n_umi": int},
        sep=sep,
    )
    matrix = frame.pivot_table(
        index="barcode", columns="guide", values="n_umi", fill_value=0
    )
    return matrix


def assignment_statistics(assignments: pd.DataFrame) -> dict:
    """
    Calculates some statistics about the assignments.

    Applies some transformations for easy json encoding
    """
    results = {}
    results["n_untested"] = int((~assignments["tested"]).sum())
    results["n_tested"] = int(assignments["tested"].sum())
    results["n_assigned"] = assignments[
        (assignments.tested) & (assignments.moi > 0)
    ].shape[0]
    results["n_unassigned"] = assignments[
        (assignments.tested) & (assignments.moi == 0)
    ].shape[0]

    mois, moi_counts = np.unique(
        assignments[(assignments.tested) & (assignments.moi > 0)].moi,
        return_counts=True,
    )
    results["dominant_moi"] = int(mois[np.argmax(moi_counts)])
    results["mois"] = list([int(x) for x in mois])
    results["moi_counts"] = list([int(x) for x in moi_counts])

    return results
