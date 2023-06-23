import numpy as np
import pandas as pd
import anndata as ad
from geomux import read_table, read_anndata, Geomux


np.random.seed(42)
N = 50
M = 5


def gen_seq(n) -> str:
    """
    creates a random sequence of size `n`
    """
    return "".join(np.random.choice(["A", "C", "T", "G"], size=n))


def create_table():
    """
    creates an arbitrary table
    """
    table = pd.DataFrame(np.random.randint(1, 10, size=(N, M))).astype(int)
    table.columns = [gen_seq(20) for _ in range(M)]
    table["barcode"] = [gen_seq(16) for _ in range(N)]
    table = table.melt(id_vars="barcode")
    table.to_csv("tests/data/table.tab", sep="\t", index=False, header=False)


def create_anndata():
    """
    creates an arbitrary anndata
    """
    mat = np.random.randint(1, 10, size=(N, M)).astype(int)
    adat = ad.AnnData(X=mat)
    adat.var.index = [gen_seq(20) for _ in range(M)]
    adat.obs.index = [gen_seq(16) for _ in range(N)]
    adat.write("tests/data/anndata.h5ad")


def test_table():
    """
    loads an arbitrary table
    """
    create_table()
    matrix = read_table("tests/data/table.tab")
    assert matrix.shape == (N, M)
    assert matrix.index.size == N
    assert matrix.columns.size == M


def test_anndata():
    """
    loads an arbitrary anndata table
    """
    create_anndata()
    matrix = read_anndata("tests/data/anndata.h5ad")
    assert matrix.shape == (N, M)
    assert matrix.index.size == N
    assert matrix.columns.size == M

def test_anndata_example():
    """
    loads an existing anndata and processes it
    """
    matrix = read_anndata("example/example.h5ad")
    gx = Geomux(
        matrix, 
        cell_names=matrix.index.values, 
        guide_names=matrix.columns.values
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
        matrix,
        cell_names=matrix.index.values,
        guide_names=matrix.columns.values
    )
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == matrix.shape[0]
