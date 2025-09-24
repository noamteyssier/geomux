import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logit
from scipy.stats import false_discovery_control, hypergeom

MIN_INSIG = 1e-10


def _test(
    matrix: csr_matrix,
    idx: np.ndarray,
    jdx: np.ndarray,
    draws: np.ndarray,
    successes: np.ndarray,
    population: int,
) -> np.ndarray:
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
    return false_discovery_control(pvalues, method="bh")


def _atomic_lor_correction(
    lbound: int,
    rbound: int,
    all_fdr: np.ndarray,
    all_assigned: np.ndarray,
    all_lor: np.ndarray,
    lor_threshold: float,
):
    """Determine whether the LOR is significant over the threshold.

    Notably this will update the `all_assigned` array and `all_lor` array **inplace**.
    """
    fdr = all_fdr[lbound:rbound]
    assigned = all_assigned[lbound:rbound]
    lors = all_lor[lbound:rbound]

    sort_idx = np.argsort(fdr)[::-1]
    min_insig = MIN_INSIG

    for sidx in sort_idx:
        lors[sidx] = logit(min_insig) - logit(fdr[sidx])
        if lors[sidx] < lor_threshold:
            assigned[sidx] = False
            min_insig = fdr[sidx]


def _lor_adjustment(
    assigned: np.ndarray,
    fdr: np.ndarray,
    idx: np.ndarray,
    lor_threshold: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    # Select all initially assigned guides
    p_assigned = assigned[assigned]

    # Initialize the LOR array
    p_lor = np.zeros(p_assigned.size)

    # Subset the FDR and cell identity arrays to initially assigned guides
    p_fdr = fdr[assigned]
    p_idx = idx[assigned]

    # Determine the cell identity step changes
    # (assumes the idx array is sorted)
    step_changes = np.concatenate(
        (np.flatnonzero(np.diff(p_idx) != 0) + 1, [p_idx.size])
    )

    # Process each cell at a time
    lbound = 0
    for rbound in step_changes:
        _atomic_lor_correction(
            lbound, rbound, p_fdr, p_assigned, p_lor, lor_threshold=lor_threshold
        )
        lbound = rbound

    adj_assigned = np.zeros_like(assigned)
    adj_assigned[np.flatnonzero(assigned)[p_assigned]] = True

    full_lor = np.zeros_like(fdr)
    full_lor[assigned] = p_lor

    return adj_assigned, full_lor


def geomux(
    matrix: csr_matrix,
    min_umi: int = 5,
    min_cells: int = 10,
    fdr_threshold: float = 0.05,
    lor_threshold: float = 10.0,
):
    # Filter out cells and guides with insufficient counts
    cell_sums = np.array(matrix.sum(axis=1)).ravel()
    guide_sums = np.array(matrix.sum(axis=0)).ravel()

    cell_mask = cell_sums >= min_cells
    guide_mask = guide_sums >= min_umi

    # Determine the relevant submatrix
    submatrix = matrix[cell_mask][:, guide_mask]

    # Determine hypergeometric parameters for the dataset
    draws = cell_sums[cell_mask]
    successes = guide_sums[guide_mask]
    population = np.sum(draws)

    # Calculate hypergeometric statistics
    idx, jdx = submatrix.nonzero()
    fdr = _test(submatrix, idx, jdx, draws, successes, population)

    # Determine the initially assigned significant set
    i_assigned = fdr <= fdr_threshold

    # Correct assignments based on log odds ratio
    assigned, lor = _lor_adjustment(
        assigned=i_assigned,
        fdr=fdr,
        idx=idx,
        lor_threshold=lor_threshold,
    )

    return (submatrix, idx, jdx, fdr, assigned, lor)
