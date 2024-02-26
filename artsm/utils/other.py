import logging
import pickle
import sys

import numpy as np


def setup_logger(name):
    """Setup a logger with a given name, usually a filename."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        stream_Handler = logging.StreamHandler()
        stream_Handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(stream_Handler)
    return logger


def mda_selection(atom_list, group='name'):
    """
    Create a selection string for MDAnalysis.

    Parameters
    ----------
    atom_list : list
        A list of atom names.
    group : str
        The group to select on. Default is 'name'.

    Returns
    -------
    str
        The selection string.

    Examples
    --------
    ['C1', 'C2', 'C3'] and 'name' -> 'name C1 or name C2 or name C3'
    """
    sel = [f'{group} {atom}' for atom in atom_list]
    sel = ' or '.join(sel)
    return sel


def serialize(*data):
    """
    Serialize data using pickle.

    If the data is None, it is not serialized.

    Parameters
    ----------
    data : tuple
        The data to serialize.

    Returns
    -------
    list
        The serialized data.
    """
    serialized = []
    for i in data:
        if i is not None:
            serialized.append(pickle.dumps(i))
        else:
            serialized.append(i)
    return serialized


def deserialize(*data):
    """
    Deserialize data using pickle.

    If the data is None, it is not deserialized and just appended to the list.

    Parameters
    ----------
        data: The data to deserialize.

    Returns
    -------
        list: The deserialized data.
    """
    deserialized = []
    for i in data:
        if i is not None:
            deserialized.append(pickle.loads(i))
        else:
            deserialized.append(i)
    return deserialized


def center_of_mass(coords, weights):
    """
    Calculate the center of mass for a set of coordinates and weights.

    Logs error if the number of coordinates is not equal to the number of weights and if not all weights are > 0.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape (N, D) representing the coordinates of N atoms in D-dimensional space.
    weights : numpy.ndarray
        Array of shape (N,) representing the weights of the N atoms.

    Returns
    -------
    numpy.ndarray
        Array of shape (D,) representing the center of mass coordinates.

    Raises:
        SystemExit: If the number of coordinates is not equal to the number of weights or if not all weights are > 0.
    """
    if coords.shape[0] != weights.size:
        logger = setup_logger(__name__)
        logger.error(f'Number of coordinates {coords.shape[0]} \u2260 {weights.size} number of weights.')
        sys.exit(-1)
    if np.sum(weights > 0.) != weights.size:
        logger = setup_logger(__name__)
        logger.error(f'Tried to calculate the center of mass, but not all weights were > 0. '
                     f'The given weights are {weights}.')
        sys.exit(-1)
    return np.sum(coords * weights.reshape((weights.size, 1)), axis=0) / np.sum(weights)


def numpy_pairwise_combinations(x):
    """Create all pairwise combinations of elements in a numpy array."""
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]
