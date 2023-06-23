import numpy as np
import pandas as pd
import anndata as ad


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


def read_anndata(filename: str) -> pd.DataFrame:
    """
    Reads an .h5ad formatted file and
    confirms that the file is in an expected format
    """
    adata = ad.read(filename)
    matrix = pd.DataFrame(
        adata.X.todense(),
        index=adata.obs.index.values,
        columns=adata.var.index.values,
    )
    return matrix
