import anndata as ad

from geomux import geomux


def test_anndata_example():
    """
    loads an existing anndata and processes it
    """
    adata = ad.read_h5ad("example/example.h5ad")
    assignments = geomux(adata)
    assert assignments.shape[0] == adata.shape[0]


def test_anndata_sparse_csr_example():
    """
    loads an existing anndata and processes it
    """
    adata = ad.read_h5ad("example/example_sparse.h5ad")
    assignments = geomux(adata)
    assert assignments.shape[0] == adata.shape[0]
