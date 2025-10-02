import numpy as np
import polars as pl


def assignment_statistics(assignments: pl.DataFrame) -> dict:
    """
    Calculates some statistics about the assignments.

    Applies some transformations for easy json encoding
    """
    results = {}
    if "tested" in assignments.columns:
        results["n_untested"] = assignments.filter(pl.col("tested").not_()).height
        results["n_tested"] = assignments.filter(pl.col("tested")).height
        results["n_assigned"] = assignments.filter(
            pl.col("tested") & pl.col("moi") > 0
        ).height
        results["n_unassigned"] = assignments.filter(
            pl.col("tested") & pl.col("moi") == 0
        ).height
        mois, moi_counts = np.unique(
            assignments.filter((pl.col("tested")) & (pl.col("moi") > 0))[
                "moi"
            ].to_numpy(),
            return_counts=True,
        )
    else:
        results["n_assigned"] = assignments.filter(pl.col("moi") > 0).height
        mois, moi_counts = np.unique(
            assignments.filter(pl.col("moi") > 0)["moi"].to_numpy(),
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
