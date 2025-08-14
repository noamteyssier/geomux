import anndata as ad
import numpy as np

from geomux import Geomux, read_table


def test_anndata_example():
    """
    loads an existing anndata and processes it
    """
    adata = ad.read_h5ad("example/example.h5ad")
    gx = Geomux(adata)
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == adata.shape[0]


def test_anndata_sparse_csr_example():
    """
    loads an existing anndata and processes it
    """
    adata = ad.read_h5ad("example/example_sparse.h5ad")
    gx = Geomux(adata)
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == adata.shape[0]


def test_table_example():
    """
    loads an existing table and processes it
    """
    matrix = read_table("example/example.tsv.gz")
    gx = Geomux(
        matrix,
        cell_names=np.array(matrix.index.values),
        guide_names=np.array(matrix.columns.values),
    )
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == matrix.shape[0]
