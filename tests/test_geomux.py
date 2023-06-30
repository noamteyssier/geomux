import numpy as np
import pandas as pd
from geomux import Geomux
from muxsim import MuxSim


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
    gx = Geomux(matrix, min_cells=1)
    gx.test()
    assignments = gx.assignments()
    assert assignments.shape[0] == n


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
    gx = Geomux(gen)
    gx.test()
    assignments = gx.assignments()
    guide_assignments = assignments.assignment.apply(
        lambda x: max(x) if len(x) > 0 else -1
    ).max()
    assert assignments.shape[0] == num_cells
    assert guide_assignments <= num_guides
    assert assignments.moi.min() >= 0
    assert (assignments.moi == 0).sum() > 0
    assert (assignments.moi == 1).sum() > 0
    assert (assignments.moi == 2).sum() > 0


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
    gx = Geomux(gen, min_cells=5)
    gx.test()
    assignments = gx.assignments()
    num_guides = np.sum(gen.sum(axis=0) >= 5)
    assert gx._n_guides == num_guides
    for a in assignments.assignment:
        for i in [0, 1, 2]:
            assert i not in a


def test_geomux_all_cells_filtered():
    """
    tests conditions where all cells are filtered
    """
    gen = np.zeros((100, 100))
    try:
        gx = Geomux(gen, min_umi=5)
        assert False
    except ValueError:
        pass


def test_geomux_all_guides_filtered():
    """
    tests conditions where all guides are filtered
    """
    gen = np.ones((100, 100))
    try:
        gx = Geomux(gen, min_cells=101)
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
    gx = Geomux(gen, min_cells=5)
    gx.test()
    assignments = gx.assignments()

    for exp, obs in zip(ms.assignments, assignments.assignment):
        if 3 in exp:
            assert 3 in obs
        elif 4 in exp:
            assert 4 in obs
