import anndata as ad
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix, csr_matrix
from sklearn import mixture
from threadpoolctl import threadpool_limits


def gaussian_mixture(
    matrix: ad.AnnData | np.ndarray | csr_matrix | csc_matrix,
    cell_names: np.ndarray | None = None,
    guide_names: np.ndarray | None = None,
    min_umi_cells: int = 3,
    n_jobs: int = -1,
) -> pl.DataFrame:
    if isinstance(matrix, ad.AnnData):
        if cell_names is None:
            cell_names = np.array(matrix.obs.index.values)
        if guide_names is None:
            guide_names = np.array(matrix.var.index.values)
        matrix = csc_matrix(matrix.X, copy=True)
    elif isinstance(matrix, np.ndarray):
        matrix = csc_matrix(matrix)
    elif isinstance(matrix, csr_matrix):
        matrix = csc_matrix(matrix, copy=True)
    elif not isinstance(matrix, csc_matrix):
        raise TypeError(
            "matrix must be an AnnData, numpy.ndarray, or scipy.sparse.csc_matrix"
        )

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

    assert isinstance(cell_names, np.ndarray), "cell_names must be a numpy.ndarray"
    assert isinstance(guide_names, np.ndarray), "guide_names must be a numpy.ndarray"
    assert isinstance(matrix, csc_matrix), "matrix must be a scipy.sparse.csc_matrix"

    return _impl_mixture(
        matrix,
        cell_names=cell_names,
        guide_names=guide_names,
        min_umi_cells=min_umi_cells,
        n_jobs=n_jobs,
    )


def _score_guide(
    matrix: csc_matrix,
    jdx: int,
    min_umi_threshold: int,
) -> np.ndarray:
    """Returns assigned cells for a given guide"""
    matrix_subset = matrix[:, jdx]

    # return early if no nonzero counts or if number of cells is less than 2
    if matrix_subset.nnz == 0 or matrix_subset.shape[0] < 2:
        return np.array([], dtype=int)

    # isolate counts
    umi_counts = matrix_subset.toarray().flatten()

    # log-transform
    log_umi_counts = np.log10(umi_counts + 1).reshape(-1, 1)

    # fit mixture model
    gmm = mixture.GaussianMixture(
        n_components=2, n_init=10, covariance_type="tied", random_state=0
    )
    gmm.fit(log_umi_counts)
    positive_component = np.argmax(gmm.means_)  # type: ignore

    # predict signal components
    mask_predicted = gmm.predict(log_umi_counts) == positive_component

    # build threshold mask
    mask_threshold = umi_counts >= min_umi_threshold

    # return all assigned cells for this guide
    return np.flatnonzero(mask_predicted & mask_threshold)


def _process_guide(
    matrix: csc_matrix,
    jdx: int,
    min_umi_threshold: int,
    cell_names: np.ndarray,
    guide_names: np.ndarray,
) -> pl.DataFrame:
    assigned_cell_indices = _score_guide(
        matrix=matrix, jdx=jdx, min_umi_threshold=min_umi_threshold
    )
    if assigned_cell_indices.size == 0:
        return pl.DataFrame({})
    else:
        return pl.DataFrame(
            {
                "cell_id": assigned_cell_indices,
                "cell": cell_names[assigned_cell_indices],
                "guide_id": jdx,
                "assignment": guide_names[jdx],
                "umi": matrix[assigned_cell_indices, jdx].data.astype(int),
            }
        )


def _process_matrix(
    matrix: csc_matrix,
    min_umi_threshold: int,
    cell_names: np.ndarray,
    guide_names: np.ndarray,
    n_jobs: int = -1,
) -> pl.DataFrame:
    num_guides = matrix.shape[1]  # type: ignore

    # Run in parallel
    with threadpool_limits(limits=1):
        assignments = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_guide)(
                matrix=matrix,
                jdx=jdx,
                min_umi_threshold=min_umi_threshold,
                cell_names=cell_names,
                guide_names=guide_names,
            )
            for jdx in range(num_guides)
        )

    try:
        assignments = pl.concat(
            [df for df in assignments if not df.is_empty()],  # type: ignore
            how="vertical_relaxed",
        )
    except ValueError:
        assignments = pl.DataFrame(
            {
                "cell_id": [],
                "cell": [],
                "guide_id": [],
                "assignment": [],
                "umi": [],
            }
        )

    return assignments


def _impl_mixture(
    matrix: csc_matrix,
    cell_names: np.ndarray,
    guide_names: np.ndarray,
    min_umi_cells: int = 3,
    n_jobs: int = -1,
) -> pl.DataFrame:
    return (
        _process_matrix(
            matrix=matrix,
            min_umi_threshold=min_umi_cells,
            cell_names=cell_names,
            guide_names=guide_names,
            n_jobs=n_jobs,
        )
        .group_by(["cell_id", "cell"])
        .agg(
            pl.col("guide_id").len().alias("moi"),
            pl.col("guide_id").str.join("|").alias("guide_ids_original"),
            pl.col("assignment").str.join("|"),
            pl.col("umi").str.join("|").alias("umis"),
        )
    )
