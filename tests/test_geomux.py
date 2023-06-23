import numpy as np
import pandas as pd
from geomux import Geomux


def sample_barcode(b_size):
    """
    generates a sample barcode
    """
    return "".join(np.random.choice(["A", "C", "T", "G"], size=b_size))


def generate_matrix(n, b_size, g_size):
    frame = pd.DataFrame(
        {
            "barcode": [sample_barcode(b_size) for _ in np.arange(n)],
            "guide": [sample_barcode(g_size) for _ in np.arange(n)],
            "n_umi": [np.random.choice(100) for _ in np.arange(n)],
        }
    )
    matrix = frame.pivot_table(index="barcode", columns="guide", values="n_umi").fillna(
        0
    )
    return matrix


def test_geomux():
    """
    tests basic usage
    """
    n = 100
    b_size = 10
    g_size = 4
    matrix = generate_matrix(n, b_size, g_size)
    gx = Geomux(matrix)
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == n
