import pandas as pd


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
