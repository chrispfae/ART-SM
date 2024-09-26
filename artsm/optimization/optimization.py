import math
import sys

from MDAnalysis.analysis import distances
import numpy as np
from scipy import optimize

from artsm.utils.angles import calc_angle, calc_dihedral
from artsm.utils.containers import element_idx
from artsm.utils.other import setup_logger


def _R(alpha, beta):
    """
    Return the rotation matrix to rotate around the x-axis by alpha and around the y-axis by beta.
    alpha and beta should be in radians.

    Parameters
    ----------
    alpha: float
        Angle to rotate around the x-axis.
    beta: float
        Angle to rotate around the y-axis.
    Returns
    -------
    numpy.ndarray
        The rotation matrix.
    """
    sin_alpha = math.sin(alpha)
    cos_alpha = math.cos(alpha)
    sin_beta = math.sin(beta)
    cos_beta = math.cos(beta)
    return np.array([[cos_beta, sin_alpha*sin_beta, sin_beta*cos_alpha],
                     [0, cos_alpha, -1*sin_alpha],
                     [(-1)*sin_beta, sin_alpha*cos_beta, cos_alpha*cos_beta]])


def _rotate_coords(coords, alpha, beta, origin=np.zeros(3)):
    """
    Rotate the given coordinates around the x-axis by alpha and around the y-axis by beta.

    -pi <= alpha/beta >= pi

    Parameters
    ----------
    coords : numpy.ndarray
        The input coordinates to be rotated.
    alpha : float
        The rotation angle around the x-axis in radians.
    beta: float
        The rotation angle around the y-axis in radians.
    origin : numpy.ndarray, default [0, 0, 0]
        The origin point around which the rotation is performed.

    Returns
    -------
    numpy.ndarray
        The rotated coordinates.

    """
    coords_mod = coords - origin
    coords_mod = coords_mod @ _R(alpha, beta).T
    coords_mod = coords_mod + origin
    return coords_mod


class EarlyStop(Exception):
    """Exception raised when Early Stop is triggered during the optimization of molecules consisting of one bead."""
    def __init__(self, xk):
        self.xk = xk


def _angle_difference(angle1, angle2):
    """
    Calculate the difference between two angles in degrees.

    Parameters
    ----------
    angle1 : float
        The first angle in radians.
    angle2 : float
        The second angle in radians.

    Returns
    -------
    float
        The difference between the two angles in degrees.
    """
    difference = abs(angle1 - angle2) * (180. / np.pi)
    if difference <= 180.:
        return difference
    else:
        return 360. - difference


def _sub_coords(idx, coords):
    """
    Calculate the Euclidean distance between two points.

    The indices are used to select the correct points from the coordinates.

    Parameters
    ----------
    idx : list
        Contains four indices [i1, j1, i2, j2].
    coords : numpy.ndarray
        A 2D arr of coordinates.

    Returns
    float
    """
    diff = coords[idx[0]][idx[1]] - coords[idx[2]][idx[3]]
    return np.linalg.norm(diff)


# Optimization functions
def _f_distance(x, coords, coms, fr_pair_idx, targets):
    """
    Calculate the objective function for the optimization of the bond lengths.

    Called by _opt_distance.
    """
    coords_rot = coords.copy()
    # Rotate fragment positions
    for idx in range(len(coords_rot)):
        coords_rot[idx] = _rotate_coords(coords_rot[idx], x[idx * 2], x[idx * 2 + 1], coms[idx])

    # Calculate current bond distances
    distances = np.apply_along_axis(_sub_coords, 1, fr_pair_idx, coords_rot)

    # Calculate objective
    obj = np.sum((distances - targets) ** 2)
    return obj


def _f_start_frs(x, coords1, com1, coords2, com2, v, targets, ref=None, ref_distance=None):
    """
    Calculate the objective function for the optimization of the first fragment pair.

    Called by _opt_start_frs.
    """
    alpha1, beta1, alpha2, beta2, shift1, shift2 = x
    db_bond_distance, db_angle1, db_angle2, db_dihedral = targets

    # rotate and shift coords
    coords1 = _rotate_coords(coords1, alpha1, beta1, com1)
    coords1 = coords1 + (v * shift1)
    coords2 = _rotate_coords(coords2, alpha2, beta2, com2)
    coords2 = coords2 + (v * shift2 * (-1))

    # Calculate objective
    obj_shift1 = abs(shift1)
    obj_shift2 = abs(shift2)
    obj_bond_distance = np.linalg.norm(coords1[1] - coords2[0]) - db_bond_distance
    angle1 = calc_angle(coords1[0], coords1[1], coords2[0])
    obj_angle1 = _angle_difference(angle1, db_angle1)
    angle2 = calc_angle(coords1[1], coords2[0], coords2[1])
    obj_angle2 = _angle_difference(angle2, db_angle2)
    dihedral = calc_dihedral(coords1[0], coords1[1], coords2[0], coords2[1])
    obj_dihedral = _angle_difference(dihedral, db_dihedral)
    objective = ((10000 * obj_shift1 ** 2) + (10000 * obj_shift2 ** 2) + (10000 * obj_bond_distance ** 2)
                 + (2 * obj_angle1 ** 2) + (2 * obj_angle2 ** 2) + (obj_dihedral ** 2))

    if ref is not None and ref_distance is not None:
        obj_ref = np.linalg.norm(coords2[2] - ref) - ref_distance
        objective += 4000 * obj_ref ** 2

    return objective


def _f_inter_frs(x, coords1, coords2, com2, v, targets, ref=None, ref_distance=None):
    """
    Calculate the objective function for the optimization of the fragment pairs that are not the starting fragment pair.

    Called by _opt_inter_frs.
    """
    alpha2, beta2, shift2 = x
    db_bond_distance, db_angle1, db_angle2, db_dihedral = targets

    # rotate and shift coords
    coords2 = _rotate_coords(coords2, alpha2, beta2, com2)
    coords2 = coords2 + (v * shift2 * (-1))

    # Calculate objective
    obj_shift = abs(shift2)
    obj_bond_distance = np.linalg.norm(coords1[1] - coords2[0]) - db_bond_distance
    angle1 = calc_angle(coords1[0], coords1[1], coords2[0])
    obj_angle1 = _angle_difference(angle1, db_angle1)
    angle2 = calc_angle(coords1[1], coords2[0], coords2[1])
    obj_angle2 = _angle_difference(angle2, db_angle2)
    dihedral = calc_dihedral(coords1[0], coords1[1], coords2[0], coords2[1])
    obj_dihedral = _angle_difference(dihedral, db_dihedral)
    objective = ((10000 * obj_shift ** 2) + (10000 * obj_bond_distance ** 2)
                 + (2 * obj_angle1 ** 2) + (2 * obj_angle2 ** 2) + (obj_dihedral ** 2))

    if ref is not None and ref_distance is not None:
        obj_ref = np.linalg.norm(coords2[2] - ref) - ref_distance
        objective += 4000 * obj_ref ** 2

    return objective


def _f_one_bead_mol(x, conf, coords_neighbors, coords_bead, box_dims):
    """
    Calculate the objective value for optimization of one bead molecules.

    The calculation of the objective value is performed only if there are neighbors, otherwise return 0.
    """
    # Rotate fragment and initialize objective
    conf_rot = _rotate_coords(conf, x[0], x[1], coords_bead)
    obj = 0.

    # Calculate objective
    dist = distances.distance_array(conf_rot, coords_neighbors, box_dims).flatten()
    if np.all(dist > 1.0):
        raise EarlyStop(x)
    obj += np.sum(np.exp((-20.) * (dist - 1.3)))

    return obj


# Update coords functions
def _update_coords_distance(molecule, x):
    """Rotate the coordinates of the individual fragments of the current molecule according to the optimization
    result in _opt_distance."""
    for idx, fr_name in enumerate(molecule.loop_order_flat):
        fr = molecule.fragments[fr_name]
        fr.pred_coords = _rotate_coords(fr.pred_coords, x[idx * 2], x[idx * 2 + 1], fr.com())


def _update_coords(molecule, x, idx):
    """Rotate the coordinates of one fragment pair according to the optimization
    result in _opt_start_frs or _opt_inter_frs."""
    fr_pair = molecule.fr_pairs[tuple(molecule.loop_order[idx])]
    fr1 = fr_pair.fr1
    fr2 = fr_pair.fr2
    v = fr2.com() - fr1.com()

    if idx == 0:
        alpha1, beta1, alpha2, beta2, shift1, shift2 = x
        fr1.pred_coords = _rotate_coords(fr1.pred_coords, alpha1, beta1, fr1.com())
        fr1.pred_coords = fr1.pred_coords + (v * shift1)
    else:
        alpha2, beta2, shift2 = x

    fr2.pred_coords = _rotate_coords(fr2.pred_coords, alpha2, beta2, fr2.com())
    fr2.pred_coords = fr2.pred_coords + (v * shift2 * (-1))


# Optimization wrappers
def _opt_info(molecule, idx, database):
    """Extract and process data of one fragment pair (specified by idx) of the current molecule to be used for
    _opt_start_frs or _opt_inter_frs."""
    fr_pair = molecule.fr_pairs[tuple(molecule.loop_order[idx])]
    fr1 = fr_pair.fr1
    fr2 = fr_pair.fr2
    idx1 = element_idx(fr_pair.con.atoms[0:2], fr1.atoms_f)
    idx2 = element_idx(fr_pair.con.atoms[2:], fr2.atoms_f)
    coords1 = fr1.pred_coords[idx1]
    coords2 = fr2.pred_coords[idx2]
    com1 = fr1.com()
    com2 = fr2.com()
    v = com2 - com1

    # Get target values from database
    bond = fr_pair.con.bond
    target_bond = database.get_bond_value(*bond)
    if target_bond is None:
        logger = setup_logger(__name__)
        logger.error(f'I could not obtain the bond distance for {bond[0]}-{bond[1]} of type {bond[2]}')
        sys.exit(-1)
    angle1 = fr_pair.con.angle1
    target_angle1 = database.get_angle_value(*angle1)
    if target_angle1 is None:
        logger = setup_logger(__name__)
        logger.error(f'I could not obtain the angle value for {angle1[0]}-{angle1[1]}-{angle1[2]} '
                     f'of type {angle1[3]} and {angle1[4]}')
        sys.exit(-1)
    angle2 = fr_pair.con.angle2
    target_angle2 = database.get_angle_value(*angle2)
    if target_angle2 is None:
        logger = setup_logger(__name__)
        logger.error(f'I could not obtain the angle value for {angle2[0]}-{angle2[1]}-{angle2[2]} '
                     f'of type {angle2[3]} and {angle2[4]}')
        sys.exit(-1)
    targets = np.array([target_bond, target_angle1, target_angle2, fr_pair.pred_dihedral])

    # Get coords of a fr3 atom that is connected to fr2
    A_fr2 = molecule.fragments_A.loc[fr2.name]
    fr_candidates = A_fr2[A_fr2 == 1].index.to_numpy()
    if fr_candidates.size == 0:
        logger = setup_logger(__name__)
        logger.error(f'Ups. Something went wrong. It seems like fragment {fr2.name} has no fragment pairs.')
        sys.exit(-1)
    elif fr_candidates.size == 1:
        ref = None
        ref_distance = None
    else:
        if fr_candidates[0] == fr1.name:
            fr3 = molecule.fragments[fr_candidates[1]]
        else:
            fr3 = molecule.fragments[fr_candidates[0]]
        fr2_atom, ref_atom = molecule.fr_pairs[(fr2.name, fr3.name)].con.atoms[1:3]
        ref = fr3.pred_coords[ref_atom == fr3.atoms_f]
        bond = molecule.fr_pairs[(fr2.name, fr3.name)].con.bond
        ref_distance = database.get_bond_value(*bond)
        if ref_distance is None:
            logger = setup_logger(__name__)
            logger.error(f'I could not obtain the bond distance for {bond[0]}-{bond[1]} of type {bond[2]}')
            sys.exit(-1)
        coords2 = np.vstack((coords2, fr2.pred_coords[fr2_atom == fr2.atoms_f]))

    return coords1, com1, coords2, com2, v, targets, ref, ref_distance


def _opt_distance(molecule, database, method, options):
    """Rotate individual fragments of the current molecule around their center of masses to obtain
    accurate bond lengths and a good starting configuration for the second optimization
    (opt_start_frs and _opt_inter_frs)."""
    dim = molecule.loop_order_flat.size
    fr_idx = dict(zip(molecule.loop_order_flat, np.arange(dim)))
    fr_pair_idx = []
    coords = [[] for _ in range(dim)]
    targets = []

    for fr_pair in molecule.fr_pairs.values():
        fr1 = fr_pair.fr1
        fr2 = fr_pair.fr2
        idx1 = fr_idx[fr1.name]
        idx2 = fr_idx[fr2.name]
        # Get coordinates of fr_pair atoms and append to coords
        coords1 = fr1.pred_coords[fr1.atoms_f == fr_pair.con.atoms[1]].flatten()
        coords2 = fr2.pred_coords[fr2.atoms_f == fr_pair.con.atoms[2]].flatten()
        coords[idx1].append(coords1)
        coords[idx2].append(coords2)
        # Get indices to correctly access them in the optimization
        fr_pair_idx.append(np.array([idx1, len(coords[idx1]) - 1, idx2, len(coords[idx2]) - 1]))

        # Get target distance from database
        bond = fr_pair.con.bond
        target = database.get_bond_value(*bond)
        if target is None:
            logger = setup_logger(__name__)
            logger.error(f'I could not obtain the bond distance for {bond[0]}-{bond[1]} of type {bond[2]}')
            sys.exit(-1)
        else:
            targets.append(target)

    # Get numpy arrays
    fr_pair_idx = np.array(fr_pair_idx)
    coords = [np.array(i) for i in coords]
    targets = np.array(targets)

    coms = np.zeros((dim, 3))
    for i, fr_name in enumerate(molecule.loop_order_flat):
        coms[i] = molecule.fragments[fr_name].com()

    start_val = np.zeros(2 * len(molecule.fragments))
    optimization = optimize.minimize(_f_distance, start_val, method=method,
                                     args=(coords, coms, fr_pair_idx, targets), options=options)
    _update_coords_distance(molecule, optimization.x)


def _opt_start_frs(molecule, database, method='L-BFGS-B', options=None):
    """
    Optimize the first fragment pair of the current molecule.

    Rotate the two fragments of the first fragment pair around their center of masses and translate them to obtain
    accurate bond lengths, angles, and dihedral angles of the connector.
    """
    coords1, com1, coords2, com2, v, targets, ref, ref_distance = _opt_info(molecule, 0, database)
    start_val = np.zeros(6)
    optimization = optimize.minimize(_f_start_frs, start_val, method=method,
                                     args=(coords1, com1, coords2, com2, v, targets, ref, ref_distance),
                                     options=options)
    _update_coords(molecule, optimization.x, 0)


def _opt_inter_frs(molecule, database, method='L-BFGS-B', options=None):
    """
    Optimize the fragment pairs (not starting fragment pair) of the current molecule.

    For each fragment pair, rotate the fragment that has not been optimized in a previous step around its center of
    mass and translate it to obtain accurate bond lengths, angles, and dihedral angles of the connectors.
    """
    for i in range(1, len(molecule.fr_pairs)):
        coords1, com1, coords2, com2, v, targets, ref, ref_distance = _opt_info(molecule, i, database)
        start_val = np.zeros(3)
        optimization = optimize.minimize(_f_inter_frs, start_val, method=method,
                                         args=(coords1, coords2, com2, v, targets, ref, ref_distance),
                                         options=options)
        _update_coords(molecule, optimization.x, i)


def optimize_molecule(molecule, database, method='L-BFGS-B', options=None):
    """
    Optimize the connectors of the current molecule such that the connecting bond lenghts, angles, and dihedral angles
    are chemically meaningful.

    Scipy optimize is used.
    First, individual fragments are rotated around their center of masses to obtain
    accurate bond lengths and a good starting configuration for the second optimization.
    Second, individual fragments are rotated around their center of masses and translated to obtain accurate
    bond lengths, angles, and dihedral angles.

    Parameters
    ----------
    molecule : Molecule
        Current molecule to be optimized.
    database : DBdata
        Database storing information of bond lengths and angles.
    method : str, default L-BFGS-B
        Optimizer method. See scipy minimize.
    options : dict, default {}
        Optimizer options. See scipy minimize.

    Returns
    -------
    numpy.ndarray
        Optimized coordinates of the current molecule.
    """
    if options is None:
        options = {}

    _opt_distance(molecule, database, method, options)
    _opt_start_frs(molecule, database, method, options)
    if len(molecule.fr_pairs) >= 2:
        _opt_inter_frs(molecule, database, method, options)


def optimize_one_bead_mol(conf, coords_neighbors, bead_coord, box_dims, method='L-BFGS-B', options=None):
    """
    Optimize the orientation of the current one bead molecules such that the distance to any neighboring atom
    is ideally more than 1 angstrom.

    Scipy optimize is used. One bead molecules are rotated around their center of mass.

    Parameters
    ----------
    conf : numpy.ndarray
        Coordinates of the one bead molecule.
    coords_neighbors : numpy.ndarray
        Coordinates of atoms close to the current one bead molecule.
    bead_coord : numpy.ndarray
        Center of mass / Coarse-grain bead position
    box_dims : numpy.ndarray
        Dimensions of the simulation box to consider periodic boundary conditions.
    method : str, default L-BFGS-B
        Optimizer method. See scipy minimize.
    options : dict, default {}
        Optimizer options. See scipy minimize.

    Returns
    -------
    numpy.ndarray
        Optimized coordinates of the current one bead molecule.
    """
    if options is None:
        options = {}

    # Optimization
    start_val = np.zeros(2)
    try:
        opt = optimize.minimize(_f_one_bead_mol, start_val, method=method,
                                args=(conf, coords_neighbors, bead_coord, box_dims), options=options)
        coords_optimized = _rotate_coords(conf, opt.x[0], opt.x[1], bead_coord)
    except EarlyStop as e:
        coords_optimized = _rotate_coords(conf, e.xk[0], e.xk[1], bead_coord)
    return coords_optimized
