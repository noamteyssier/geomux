from geomux import read_table, read_anndata, Geomux


def test_anndata_example():
    """
    loads an existing anndata and processes it
    """
    matrix = read_anndata("example/example.h5ad")
    gx = Geomux(
        matrix, cell_names=matrix.index.values, guide_names=matrix.columns.values
    )
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == matrix.shape[0]


def test_anndata_sparse_csr_example():
    """
    loads an existing anndata and processes it
    """
    matrix = read_anndata("example/example_sparse.h5ad")
    gx = Geomux(
        matrix, cell_names=matrix.index.values, guide_names=matrix.columns.values
    )
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == matrix.shape[0]


def test_table_example():
    """
    loads an existing table and processes it
    """
    matrix = read_table("example/example.tsv.gz")
    gx = Geomux(
        matrix, cell_names=matrix.index.values, guide_names=matrix.columns.values
    )
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == matrix.shape[0]
