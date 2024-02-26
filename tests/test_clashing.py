import numpy as np

from artsm.utils.clashing_atoms import clashing_atoms
from fixtures import args_clashing


def test_clashing_atoms(args_clashing):
    rng = np.random.default_rng(seed=45)
    box_dims, coords1, coords2, coords3 = args_clashing
    clashing_atoms(coords1, box_dims, rng)
    expected1 = np.array([[-0.06635556,  0.01236139,  0.01236139],
                          [0.12523564, -0.00354765, -0.00354765],
                          [0.37592784, -0.0017461, -0.0017461],
                          [0.57753137, 0.00527165, 0.00527165]])
    assert np.allclose(coords1, expected1)
    clashing_atoms(coords2, box_dims, rng)
    assert not np.allclose(coords2, [1.543543, 0., 0.])
    clashing_atoms(coords3, box_dims, rng)
    expected2 = np.array([[-0.08760787, -0.0313822, -0.01794145],
                 [0.07383858, 0.0042369, -0.01240138],
                 [0.31467828, 0.07757383, 0.02089401],
                 [0.47888982, 0.12937028, 0.04581764]])
    assert np.allclose(coords3, expected2)
