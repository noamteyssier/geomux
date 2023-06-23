import numpy as np
import pandas as pd
from geomux import Geomux


def sample_barcode(b_size):
    """
    generates a sample barcode
    """
    return "".join(np.random.choice(["A", "C", "T", "G"], size=b_size))


def test_geomux():
    """
    tests basic usage
    """
    n = 1000
    b_size = 20
    g_size = 4
    frame = pd.DataFrame(
        {
            "barcode": [sample_barcode(b_size) for _ in np.arange(n)],
            "guide": [sample_barcode(g_size) for _ in np.arange(n)],
            "n_umi": [np.random.choice(100) for _ in np.arange(n)],
        }
    )
    geom = Geomux()
    geom.fit(frame)
    geom.predict()
    geom.assignments()
