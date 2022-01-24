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
            sep=sep)
    return frame


def read_anndata(filename: str) -> pd.DataFrame:
    """
    Reads an .h5ad formatted file and
    confirms that the file is in an expected format
    """
    frame = ad.read(filename).to_df().reset_index()
    frame = frame\
        .melt(
            id_vars=frame.columns.values[0],
            var_name="guide",
            value_name="n_umi")
    frame = frame[frame.n_umi > 0]
    frame.columns = ["barcode", "guide", "n_umi"]
    return frame
