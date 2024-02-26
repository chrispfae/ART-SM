import itertools
import sys

import numpy as np
import pandas as pd

from artsm.model.mainconfs import metric_dihedral_angle
from artsm.model.features import _stack_X
from artsm.optimization.optimization import _rotate_coords
from artsm.utils.other import mda_selection, setup_logger


def _generate_combinations(*ints):
    r"""
    Generate all one-hot encodings and their combinations given multiple integer values.

    The integer values specify the number of classes. For instance 3 indicates that the feature has 3 classes and thus
    the one-hot-encodings [1, 0, 0], [0, 1, 0], and [0, 0, 1]. Afterwards, all combinations of one-hot encodings are
    generated.

    Parameters
    ----------
    *ints : tuple of int
        Integers indicating the number of classes per feature.

    Returns
    -------
    numpy.ndarray
        All combinations of possible one-hot encodings.

    Examples
    --------
    Given are the integers 3 and 2. Thus, the one-hot encodings are [1, 0, 0], [0, 1, 0], [0, 0, 1] and [1, 0], [0, 1].
    The resulting combinations (concatenations) are [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0], [0, 0, 1, 0, 1].
    """
    if len(ints) < 2:
        logger = setup_logger(__name__)
        logger.error('At least two arguments have to be given.')
        sys.exit(-1)
    X = None
    for section in ints:
        if X is None:
            X = np.diag(np.ones(section, dtype=np.int16))
        else:
            dim_x = X.shape[0] * section
            dim_y = X.shape[1] + section
            T = np.zeros((dim_x, dim_y), dtype=np.uint8)
            X_add = np.diag(np.ones(section, dtype=np.uint8))
            generator = itertools.product(X, X_add)
            for count, iterable in enumerate(generator):
                T[count, :] = np.array(list(itertools.chain.from_iterable(iterable)))
            X = T.copy()
    return X


def _derive_conformation_indices(arr, *lengths):
    """
    Reverse one-hot encoding given concatenated one-hot encodings.

    Given an array of concatenated one-hot encodings, e.g. [1, 0, 0, 0, 1, 0, 1], the class labels are derived.
    The lengths arguments specify the number of elements per one-hot encoding in order. So 3, 2, 2 for the array
    [1, 0, 0, 0, 1, 0, 1] means that the first 3 elements [1, 0, 0] belong to feature one, the following two [0, 1] to
    feature two, and [0, 1] to feature three. Afterwards, the one-hot encodings are reversed, which results in [0, 1, 1]
    for the given example.

    Parameters
    ----------
    arr : Union(list, numpy.ndarray) of zeros and ones
        Concatenated one-hot encoded features.
    *lengths : tuple of int
        Length of each one-hot encoded feature.
    Returns
    -------
    list of int
        Decoded one-hot encodings.
    """
    confs = []
    lb = 0
    for length in lengths:
        ub = lb + length
        pos = np.where(arr[lb:ub] == 1)[0][0]
        confs.append(pos)
        lb = ub
    return confs


def _extract_features(fr_pair, residue):
    """
    Extract features from the given fragment pair and coarse-grained residue.

    Currently, the only feature is the distance between the two coarse-grained beads.

    Parameters
    ----------
    fr_pair : FragmentPair
    residue : MDAnalysis.core.groups.AtomGroup

    Returns
    -------
    numpy.ndarray
    Feature array.
    """
    fr1_simulation = residue.atoms.select_atoms(mda_selection([fr_pair.fr1.name]))
    fr2_simulation = residue.atoms.select_atoms(mda_selection([fr_pair.fr2.name]))
    com = np.linalg.norm(fr1_simulation.positions - fr2_simulation.positions)
    return np.array([com])


def _nearest_neighbor_idx(main_internal, pred_internal):
    """
    Find the best matching main conformation.

    All the main conformations of a fragment pair are stored in main_internal. They are compared to the main
    conformation given in pred_internal. The index of the best matching main conformation in main_internal in terms of
    the metric 'metric_dihedral_angle) is returned.

    Parameters
    ----------
    main_internal : numpy.ndarray
        Array of shape (n, m), where n is the number of main conformations and m the number of dihedral angles.
    pred_internal : numpy.ndarray
        Array of shape (, m), where m is the number of dihedral angles.

    Returns
    -------
    int: Index of the best matching main conformation in `main_internal` for the given main conformation `pred_internal`.
    """
    dist = np.array([metric_dihedral_angle(pred_internal, d_angles) for d_angles in main_internal])
    min_dist_idx = np.argmin(dist)
    return min_dist_idx


def preproc(fr_pair, residue, pred_internal1=None):
    """
    Generate the feature vectors for a given fragment pair and coarse-grained residue.
    Parameters
    ----------
    fr_pair : FragmentPair
    residue : MDAnalysis.core.groups.AtomGroup
        Residue of the current coarse-grained snapshot.
    pred_internal1: numpy.ndarray
        Dihedral angles of the main conformation of an already predicted fragment.

    Returns
    -------
    pandas.DataFrame
        Feature vectors for the current fragment pair and coarse-grained residue.
        Index: Range index
        Columns: Fragment1 Fragment2 Connector COM-distance (first three are one-hot encoded)
    """
    main_coords1, main_internal1, main_coords2, _, main_dihedrals, model = fr_pair.get_models()
    n_cluster1 = main_coords1.shape[0]
    n_cluster2 = main_coords2.shape[0]
    n_clusterd = main_dihedrals.shape[0]

    if fr_pair.reverse:
        X = _generate_combinations(n_cluster2, n_cluster1, n_clusterd)
    else:
        X = _generate_combinations(n_cluster1, n_cluster2, n_clusterd)

    if pred_internal1 is not None:
        column = np.zeros((X.shape[0], main_coords1.shape[0]))  # Due to loop order, only the label of fr1 can be known
        idx = _nearest_neighbor_idx(main_internal1, pred_internal1)
        column[:, idx] = 1
        if fr_pair.reverse:
            X[:, n_cluster2:(n_cluster2+n_cluster1)] = column
        else:
            X[:, 0:n_cluster1] = column

    features = _extract_features(fr_pair, residue)
    features.shape = (1, features.size)
    features = np.repeat(features, X.shape[0], axis=0)
    X = _stack_X(X, features)
    X = pd.DataFrame(X, columns=model.feature_names_in_)
    return X


def prediction(fr_pair, X, residue, rng):
    """
    Predict the probabilities for the given feature vectors, sample a combination of main conformations according to
    these probabilities, and return the main conformations.
    Parameters
    ----------
    fr_pair : FragmentPair
    X : pandas.Dataframe
        Feature vectors.
        Index: Range index
        Columns: Fragment1 Fragment2 Connector COM-distance (first three are one-hot encoded)
    residue : MDAnalysis.core.groups.AtomGroup
        Residue of the current coarse-grained snapshot.
    rng : np.random.default_rng()
        Default random number generator of numpy.

    Returns
    -------
    tuple
        coords and dihedral angles of main conformations of fragment 1 and 2, and the dihedral angle for the connector.
    """
    main_coords1, main_internal1, main_coords2, main_internal2, main_dihedrals, model = fr_pair.get_models()

    probabilities = model.predict(X)
    probabilities[probabilities < 0] = 0.
    if np.sum(probabilities) == 0.:
        logger = setup_logger(__name__)
        logger.warning(f'Probabilities can not be predicted for residue {residue.resnum}.'
                       f' Main conformations will be randomly determined.')
        probabilities = np.repeat(1. / probabilities.size, probabilities.size)
    probabilities = probabilities / sum(probabilities)
    sampling = rng.choice(a=probabilities.size, size=1, p=probabilities)[0]
    conf = X.iloc[sampling].to_numpy()

    if fr_pair.reverse:
        indices = _derive_conformation_indices(conf, main_coords2.shape[0], main_coords1.shape[0],
                                               main_dihedrals.shape[0])
        fr1_coords = main_coords1[indices[1]]
        fr1_internal = main_internal1[indices[1]]
        fr2_coords = main_coords2[indices[0]]
        fr2_internal = main_internal2[indices[0]]
    else:
        indices = _derive_conformation_indices(conf, main_coords1.shape[0], main_coords2.shape[0],
                                               main_dihedrals.shape[0])
        fr1_coords = main_coords1[indices[0]]
        fr1_internal = main_internal1[indices[0]]
        fr2_coords = main_coords2[indices[1]]
        fr2_internal = main_internal2[indices[1]]

    dihedral = main_dihedrals[indices[2]]
    return fr1_coords, fr1_internal, fr2_coords, fr2_internal, dihedral


def rotate_random(coords, com, rng):
    """
    Rotate the given coordinates randomly around the center of mass.

    The rotation angles are uniformly distributed between 0 and 2 * pi. The rotation is performed using the
    `_rotate_coords` function.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates to be rotated.
    com : tuple
        Center of mass coordinates (x, y, z).

    Returns
    -------
    numpy.ndarray
        Rotated coordinates.
    """
    alpha = rng.uniform(0, 2 * np.pi)
    beta = rng.uniform(0, 2 * np.pi)
    return _rotate_coords(coords, alpha, beta, com)
