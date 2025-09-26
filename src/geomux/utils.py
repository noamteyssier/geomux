import numpy as np
import pandas as pd
import polars as pl


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


def assignment_statistics(assignments: pl.DataFrame) -> dict:
    """
    Calculates some statistics about the assignments.

    Applies some transformations for easy json encoding
    """
    results = {}
    results["n_untested"] = assignments.filter(pl.col("tested").not_()).height
    results["n_tested"] = assignments.filter(pl.col("tested")).height
    results["n_assigned"] = assignments.filter(
        pl.col("tested") & pl.col("moi") > 0
    ).height
    results["n_unassigned"] = assignments.filter(
        pl.col("tested") & pl.col("moi") == 0
    ).height
    mois, moi_counts = np.unique(
        assignments.filter((pl.col("tested")) & (pl.col("moi") > 0))["moi"].to_numpy(),
        return_counts=True,
    )

    # Set default values if no assignments
    if mois.size == 0:
        results["dominant_moi"] = 0
        results["mois"] = []
        results["moi_counts"] = []
        return results

    results["dominant_moi"] = int(mois[np.argmax(moi_counts)])
    results["mois"] = list([int(x) for x in mois])
    results["moi_counts"] = list([int(x) for x in moi_counts])

    return results
