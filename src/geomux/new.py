import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.special import logit
from scipy.stats import false_discovery_control, hypergeom

MAX_INSIG = 1 - 1e-10


def _test(
    matrix: csr_matrix,
    idx: np.ndarray,
    jdx: np.ndarray,
    draws: np.ndarray,
    successes: np.ndarray,
    population: int,
) -> tuple[np.ndarray, np.ndarray]:
    # non-zero values
    values = matrix[idx, jdx]

    k = values - 1
    M = population
    n = successes[jdx]
    N = draws[idx]

    pvalues = hypergeom.sf(k, M, n, N).flatten()
    pvalues = np.clip(
        pvalues,
        np.min(pvalues[pvalues > 0]),  # set zeros to minimum nonzero p-value
        1.0,  # clip to 1.0 just in case
    )

    return pvalues, false_discovery_control(pvalues, method="bh")


def _atomic_lor_correction(
    lbound: int,
    rbound: int,
    fdr: np.ndarray,
    assigned: np.ndarray,
    lors: np.ndarray,
    lor_threshold: float,
    max_insig: float = MAX_INSIG,
):
    """Determine whether the LOR is significant over the threshold.

    Notably this will update `assigned` and `lors` arrays **inplace**.
    """
    sub_fdr = fdr[lbound:rbound]
    sub_assigned = assigned[lbound:rbound]
    sub_lors = lors[lbound:rbound]

    # Skip full insignificant sets
    if not np.any(sub_assigned):
        return

    # Set the initial minimum insignificant FDR
    if np.all(sub_assigned):
        min_insig = max_insig
    else:
        min_insig = min(np.min(sub_fdr[~sub_assigned]), max_insig)

    # Sort the FDR in descending order (most insignificant first)
    sort_idx = np.argsort(sub_fdr)[::-1]

    # Iterate over the sorted FDR values
    for s in sort_idx:
        # skip insignificant
        if not sub_assigned[s]:
            continue

        # Calculate the LOR
        s_fdr = sub_fdr[s]
        sub_lors[s] = logit(min_insig) - logit(s_fdr)

        # Reset the minimum insignificant FDR if the LOR is insignificant
        if sub_lors[s] < lor_threshold:
            sub_assigned[s] = False
            min_insig = min(s_fdr, max_insig)


def _lor_adjustment(
    assigned: np.ndarray,
    fdr: np.ndarray,
    idx: np.ndarray,
    lor_threshold: float = 10.0,
) -> np.ndarray:
    step_changes = np.concatenate((np.flatnonzero(np.diff(idx) != 0) + 1, [idx.size]))
    lors = np.zeros_like(fdr)

    lbound = 0
    for rbound in step_changes:
        _atomic_lor_correction(
            lbound=lbound,
            rbound=rbound,
            fdr=fdr,
            assigned=assigned,
            lors=lors,
            lor_threshold=lor_threshold,
            max_insig=fdr.mean(),
        )
        lbound = rbound

    return lors


def _build_results(
    matrix: csr_matrix,
    idx: np.ndarray,
    jdx: np.ndarray,
    cell_names: np.ndarray,
    cell_mask: np.ndarray,
    guide_names: np.ndarray,
    guide_mask: np.ndarray,
    total_umis: np.ndarray,
    assigned: np.ndarray,
    fdrs: np.ndarray,
    lors: np.ndarray,
) -> pl.DataFrame:
    # Get tested cell info and original indices
    tested_cell_names = cell_names[cell_mask]
    tested_n_umis = total_umis[cell_mask]

    # Create mapping from submatrix indices to original indices
    original_cell_indices = np.flatnonzero(cell_mask)
    original_guide_indices = np.flatnonzero(guide_mask)

    # Build assigned dataframe
    assigned_idx = idx[assigned]  # submatrix cell indices
    assigned_jdx = jdx[assigned]  # submatrix guide indices
    assigned_counts = np.array(matrix[assigned_idx, assigned_jdx]).flatten()

    assigned_df = (
        pl.DataFrame(
            {
                "cell_id": original_cell_indices[assigned_idx],
                "submatrix_id": assigned_idx,
                "cell": tested_cell_names[assigned_idx],
                "guide_id_submatrix": assigned_jdx,
                "guide_id_original": original_guide_indices[assigned_jdx],
                "assignment": guide_names[guide_mask][assigned_jdx],
                "umis": assigned_counts.astype(str),
                "fdr": fdrs[assigned].astype(str),
                "log_odds": lors[assigned].astype(str),
                "n_umi": tested_n_umis[assigned_idx],
            }
        )
        .group_by(["cell_id", "submatrix_id", "cell"])
        .agg(
            pl.col("assignment").len().alias("moi"),
            pl.col("n_umi").max(),
            pl.col("assignment").str.join("|"),
            pl.col("guide_id_original").str.join("|").alias("guide_ids_original"),
            pl.col("umis").str.join("|"),
            pl.col("fdr").str.join("|"),
            pl.col("log_odds").str.join("|"),
        )
        .with_columns(pl.lit(True).alias("tested"))
    )

    # For unassigned cells
    all_tested_cells = set(range(len(tested_cell_names)))
    assigned_cell_set = set(assigned_idx)
    unassigned_cell_indices = np.array(list(all_tested_cells - assigned_cell_set))

    unassigned_df = pl.DataFrame(
        {
            "cell_id": original_cell_indices[unassigned_cell_indices],
            "submatrix_id": unassigned_cell_indices,
            "cell": tested_cell_names[unassigned_cell_indices],
            "moi": 0,
            "n_umi": tested_n_umis[unassigned_cell_indices],
            "assignment": "",
            "guide_ids_original": "",
            "umis": "",
            "fdr": "",
            "log_odds": "",
            "tested": True,
        }
    )

    # For untested cells - these keep their original indices
    untested_indices = np.where(~cell_mask)[0]
    untested_cell_names = cell_names[~cell_mask]
    untested_n_umis = total_umis[~cell_mask]

    missing_df = pl.DataFrame(
        {
            "cell_id": untested_indices,
            "submatrix_id": np.full(len(untested_cell_names), np.nan),
            "cell": untested_cell_names,
            "moi": 0,
            "n_umi": untested_n_umis,
            "assignment": "",
            "guide_ids_original": "",
            "umis": "",
            "fdr": "",
            "log_odds": "",
            "tested": False,
        }
    )

    return pl.concat([assigned_df, unassigned_df, missing_df], how="vertical_relaxed")


def geomux(
    matrix: csr_matrix,
    cell_names: np.ndarray | None = None,
    guide_names: np.ndarray | None = None,
    min_umi_cells: int = 5,
    min_umi_guides: int = 10,
    fdr_threshold: float = 0.05,
    lor_threshold: float = 50.0,
    subtract: bool = True,
) -> pl.DataFrame:
    if cell_names is not None:
        if cell_names.size != matrix.shape[0]:  # type: ignore
            raise ValueError(
                f"cell_names (len={cell_names.size}) must have the same length as the number of rows in the matrix ({matrix.shape[0]})"  # type:ignore
            )
    else:
        cell_names = np.arange(matrix.shape[0])  # type: ignore
    if guide_names is not None:
        if guide_names.size != matrix.shape[1]:  # type: ignore
            raise ValueError(
                f"guide_names (len={guide_names.size}) must have the same length as the number of columns in the matrix ({matrix.shape[1]})"  # type:ignore
            )
    else:
        guide_names = np.arange(matrix.shape[1])  # type: ignore

    assert isinstance(cell_names, np.ndarray)
    assert isinstance(guide_names, np.ndarray)

    # Filter out cells and guides with insufficient counts
    print("=== Filtering ===")
    cell_sums = np.array(matrix.sum(axis=1)).ravel()
    guide_sums = np.array(matrix.sum(axis=0)).ravel()

    cell_mask = cell_sums >= min_umi_cells
    guide_mask = guide_sums >= min_umi_guides

    # Determine the relevant submatrix
    submatrix = matrix[cell_mask][:, guide_mask]
    if subtract:
        submatrix.data -= 1
        submatrix.eliminate_zeros()

    # Determine hypergeometric parameters for the dataset
    draws = np.array(submatrix.sum(axis=1)).ravel()
    successes = np.array(submatrix.sum(axis=0)).ravel()
    population = submatrix.sum()

    print(f">> Number of testable cells: {cell_mask.sum()}")
    print(f">> Number of testable guides: {guide_mask.sum()}")
    print(f">> Mean cell UMI: {draws.mean():.2f}")
    print(f">> Mean guide UMI: {successes.mean():.2f}")
    print(f">> Total UMIs: {population}")

    # Calculate hypergeometric statistics
    print("=== Hypergeometric Test ===")
    idx, jdx = submatrix.nonzero()
    pvalues, fdr = _test(submatrix, idx, jdx, draws, successes, population)

    # Determine the initially assigned significant set
    assigned = fdr <= fdr_threshold
    print(f">> Initially assigned cell-guide pairs: {assigned.sum()}")

    # Correct assignments based on log odds ratio
    print("=== Log Odds Ratio Adjustment ===")
    lor = _lor_adjustment(
        assigned=assigned,
        fdr=fdr,
        idx=idx,
        lor_threshold=lor_threshold,
    )
    print(f">> Final assigned cell-guide pairs: {assigned.sum()}")

    results = _build_results(
        matrix=submatrix,
        idx=idx,
        jdx=jdx,
        cell_names=cell_names,
        cell_mask=cell_mask,
        guide_names=guide_names,
        guide_mask=guide_mask,
        total_umis=cell_sums,
        assigned=assigned,
        fdrs=fdr,
        lors=lor,
    )

    return results
