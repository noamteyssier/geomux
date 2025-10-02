import numpy as np
import pandas as pd
from muxsim import MuxSim
from scipy.sparse import csr_matrix

from geomux import geomux, gaussian_mixture


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
    matrix_csr = csr_matrix(matrix.values)
    cell_names = np.array(matrix.index.values)
    guide_names = np.array(matrix.columns.values)
    assignments = geomux(
        matrix_csr, cell_names=cell_names, guide_names=guide_names, min_umi_cells=1
    )
    assert assignments.shape[0] == n


def test_mixture():
    """
    tests basic usage
    """
    n = 100
    b_size = 10
    g_size = 4
    matrix = generate_matrix(n, b_size, g_size)
    matrix_csr = csr_matrix(matrix.values)
    cell_names = np.array(matrix.index.values)
    guide_names = np.array(matrix.columns.values)
    assignments = gaussian_mixture(
        matrix_csr, cell_names=cell_names, guide_names=guide_names, min_umi_cells=1
    )
    assert assignments.height > 0  # found some assignments


def test_assignments():
    """
    tests assignments with easy data
    """
    num_cells = 1000
    num_guides = 100
    ms = MuxSim(
        num_cells=num_cells,
        num_guides=num_guides,
        n=20,
    )
    gen = ms.sample()
    assignments = geomux(gen)
    assert assignments.shape[0] == num_cells
    assert assignments["moi"].min() >= 0
    assert (assignments["moi"] == 0).sum() > 0
    assert (assignments["moi"] == 1).sum() > 0
    assert (assignments["moi"] == 2).sum() > 0


def test_geomux_min_cells():
    """
    tests min_cells
    """
    num_cells = 1000
    num_guides = 100
    ms = MuxSim(
        num_cells=num_cells,
        num_guides=num_guides,
        n=20,
    )
    gen = ms.sample()
    gen[:, :3] = 0
    assignments = geomux(gen, min_umi_guides=5)
    num_guides = np.sum(gen.sum(axis=0) >= 5)
    # Check that filtered guides don't appear in assignments
    for a in assignments["assignment"]:
        if a:  # Skip empty assignments
            items = a.split("|")
            for i in [0, 1, 2]:
                assert str(i) not in items


def test_geomux_all_cells_filtered():
    """
    tests conditions where all cells are filtered
    """
    gen = np.zeros((100, 100))
    try:
        _assignments = geomux(gen, min_umi_cells=5)
        assert False
    except ValueError:
        pass


def test_geomux_all_guides_filtered():
    """
    tests conditions where all guides are filtered
    """
    gen = np.ones((100, 100))
    try:
        _assignments = geomux(gen, min_umi_guides=101)
        assert False
    except ValueError:
        pass


def test_geomux_correct_assignment():
    """
    tests that the correct assignment is made for certain cases
    """
    num_cells = 1000
    num_guides = 100
    ms = MuxSim(
        num_cells=num_cells,
        num_guides=num_guides,
        n=20,
    )
    gen = ms.sample()
    gen[:, :3] = 0
    assignments = geomux(gen, min_umi_guides=5)

    # Create a mapping from assignment cell indices to original cell indices
    assignment_dict = {}
    for i, row in enumerate(assignments.iter_rows(named=True)):
        assignment_dict[row["cell_id"]] = row["assignment"]

    for cell_idx, exp in enumerate(ms.assignments):
        if cell_idx in assignment_dict:
            obs = assignment_dict[cell_idx]
            if obs:  # Skip empty assignments
                items = obs.split("|")
                if 3 in exp:
                    # Guide 3 should still be in the filtered guides since guides 0,1,2 were set to 0
                    # The guide names in assignments are the original guide names
                    assert "3" in items
