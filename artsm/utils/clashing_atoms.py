from collections import defaultdict
import importlib

from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.util import check_box
import numpy as np

from artsm.utils.other import setup_logger

_distances = {'serial': importlib.import_module(".c_distances",
                                                package="MDAnalysis.lib")}


def _get_dist_func(box_dims=None):
    """
    Determine the correct distance function from MDAnalysis to include periodic boundary conditions.
    Parameters
    ----------
    box_dims : np.ndarray
        Simulation box.
    Returns
    -------
    tuple
        func
            Distance function.
        np.ndarray
            The x, y, and z dimensions of the box without the angles.
    """
    if box_dims is None:
        return getattr(_distances['serial'], 'calc_distance_array'), None
    else:
        boxtype, box = check_box(box_dims)
        if boxtype == 'ortho':
            return getattr(_distances['serial'], 'calc_distance_array_ortho'), box
        else:
            return getattr(_distances['serial'], 'calc_distance_array_triclinic'), box


def _clashing_atoms_fastns(coords, radius=0.15, box_dims=None, ref=None):
    """
    Find clashing atoms, i.e. atoms that are closer than radius (default=0.15) angstrom.

    If a reference atom is given (ref), clashes between the reference atom and the given coords are determined.
    Otherwise, coords are compared with themselves.
    Distances are calculated by including periodic boundary conditions.
    This uses the Neighbor search library from MDAnalysis.
    It is a serialized Cython version greatly inspired by the NS grid search
    implemented in GROMACS.

    Parameters
    ----------
    coords : numpy.ndarray
        Atom coordinates.
    radius : float, default 0.15
        Distance at which atoms are considered clashing.
    box_dims : numpy.ndarray, default None
        Box dimensions of the simulation cell.
    ref : numpy.ndarray
        Reference atom coordinates.
    Returns
    -------
    ndarray
        The indices of the clashing atoms in the atom coordinates array.
    """
    coords_float = coords.copy().astype(np.float32)

    if box_dims is not None:
        nb_search = FastNS(radius, coords_float, box_dims.astype(np.float32), pbc=True)
    else:
        logger = setup_logger(__name__)
        logger.warning('Input pdb does not contain information on the simulation box. '
                       'Orthogonal Pseudo-box will be constructed. Periodic boundary conditions will be not considered.')
        lmax = coords.max(axis=0)
        lmin = coords.min(axis=0)
        pseudobox = np.empty(6)
        pseudobox[:3] = 1.1 * (lmax - lmin)
        pseudobox[3:] = 90.
        shift = coords.copy()
        shift -= lmin
        nb_search = FastNS(radius, shift, box=pseudobox, pbc=False)

    if ref is None:
        nb_results = nb_search.self_search()
    else:
        ref_float32 = ref.copy().astype(np.float32)
        nb_results = nb_search.search(ref_float32)
    nb_pairs = nb_results.get_pairs()
    return nb_pairs


def _shift_atoms(coords, idx_pairs, rng):
    """
    Shift atoms to avoid clashes.

    For each pair (a1, a2) the vector v = a1 - a2 / |a1 - a2| is calculated. a1_new = a1 + 0.075 * v.
    a2_new = a2 - 0.075 * v, i.e. atoms are shifted away from other by 0.075 angstrom.
    Additionally, a1 and a2 are randomly shifted by a small amount to avoid any special geometric traps.
    Each atom is shifted only once, i.e. given the pairs (1, 4) and (1, 8), only the pair (1, 4) is considered.

    Parameters
    ----------
    coords: numpy.ndarray
        Atom coordinates.
    idx_pairs: numpy.ndarray
        Indices of atom pairs that should be shifted.
    """

    seen = []
    for a1, a2 in idx_pairs:
        if a1 in seen or a2 in seen:
            continue
        seen.append(a1)
        seen.append(a2)

        # Shift a1 and a2 away from each other.
        v = coords[a1] - coords[a2]
        if not np.allclose(v, [0, 0, 0]):
            v_norm = v / np.linalg.norm(v)
            coords[a1] += v_norm * 0.075
            coords[a2] -= v_norm * 0.075
        # Random shift to avoid special geometric traps.
        coords[a1] += rng.standard_normal() * 0.01
        coords[a2] += rng.standard_normal() * 0.01


def clashing_atoms(coords, box_dims, rng, radius=0.15):
    """
    Find and shift atoms that are closer than radius (default=0.15) angstrom.
    Parameters
    ----------
    coords : numpy.ndarray
        Atom coordinates.
    box_dims : numpy.ndarray, default None
        Box dimensions of the simulation cell.
    rng : np.random.default_rng()
        Default random number generator of numpy.
    radius : float, default 0.15
        Distance at which atoms are considered clashing.
    """
    clashing_atom_pairs = _clashing_atoms_fastns(coords, radius, box_dims)
    while clashing_atom_pairs.size > 0:
        _shift_atoms(coords, clashing_atom_pairs, rng)
        # Get clashing atoms
        idx = np.unique(clashing_atom_pairs.flatten())
        clashing_atom_pairs_new = _clashing_atoms_fastns(coords, radius, box_dims=box_dims, ref=coords[idx])
        clashing_atom_pairs_new = [[idx[i], j] for i, j in clashing_atom_pairs_new]
        clashing_atom_pairs = np.array([[i, j] for i, j in clashing_atom_pairs_new if i != j])


def find_neighbors(coords, box_dims, threshold=10):
    """
    For each atom, find the at least `threshold` neighbors.
    Parameters
    ----------
    coords : numpy.ndarray
        Atom coordinates.
    box_dims : numpy.ndarray
        Box dimensions of the simulation cell.
    threshold : int, default 10
        The minimum number of neighbors to find for each atom.
    Returns
    -------
    dict
        A dictionary containing for each atom a list of neighbor atoms.
        The number of neighbors can be much more than the specified threshold.
    """
    # Adjust the threshold if the number of atoms is smaller than the threshold.
    if coords.shape[0] - 1 < threshold:
        threshold = coords.shape[0]

    # Determine for each atom all neighbors within 10 angstrom.
    # If any atom does not have at least `threshold` atoms, increase the considered distance by 5 angstrom.
    # Repeat this process.
    r = 5.
    r_to_low = True
    while r_to_low:
        r += 5.
        neighbor_pairs = _clashing_atoms_fastns(coords, radius=r, box_dims=box_dims)
        neighbors = defaultdict(list)
        for i, j in neighbor_pairs:
            neighbors[i].append(j)
            neighbors[j].append(i)
        r_to_low = False
        for _, j in neighbors.items():
            if len(j) < threshold:
                r_to_low = True
                break
    return neighbors
