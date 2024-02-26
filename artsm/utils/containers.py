from collections import defaultdict
import logging
import sys

import numpy as np
import re

from artsm.utils.other import setup_logger


def idx_atoms_f(atom_list):
    """Return boolean array indicating the positions of all heavy atoms (not hydrogens) in the given array."""
    idx = np.zeros(atom_list.size, dtype=bool)
    for i, atom in enumerate(atom_list):
        if not re.match(r'^H$', atom) and (not re.match(r'^H.+', atom) or re.match(r'H[EOS]$', atom)):
            idx[i] = True
    return idx


def extract_atoms_f(atom_list):
    """Return all heavy atoms (not hydrogens) in the given array."""
    atoms_f = [atom for atom in atom_list if not re.match(r'^H$', atom)]
    atoms_f = [atom for atom in atoms_f if (not re.match(r'^H.+', atom) or re.match(r'H[EOS]$', atom))]
    return np.array(atoms_f)


def idx_atoms_h(atom_list):
    """Return boolean array indicating the positions of all hydrogen atoms."""
    idx = np.zeros(atom_list.size, dtype=bool)
    for i, atom in enumerate(atom_list):
        if re.match(r'^H$', atom) or (re.match(r'^H.+', atom) and not re.match(r'H[EOS]$', atom)):
            idx[i] = True
    return idx


def extract_atoms_h(atom_list):
    """
    Return all hydrogen atoms in the given array.

    Atoms that start with H and do not end with E, O or S are considered hydrogen atoms.
    """
    atoms_f = [atom for atom in atom_list if re.match(r'^H$', atom) or (re.match(r'^H.+', atom) and not re.match(r'H[EOS]$', atom))]
    return np.array(atoms_f)


def type_random_value(dictionary, rng):
    """Return the type of random value in a dictionary."""
    random_value = rng.choice(list(dictionary.values()), size=1)
    return type(random_value)


def remove_duplicates(array_):
    """Remove duplicates from a numpy array."""
    _, idxs = np.unique(array_, return_index=True)
    return array_[np.sort(idxs)]


def element_idx(array1, array2):
    """
    Determines the positions of the elements of array1 in the order they are present in array2.

    array1.size <= array2.size

    Examples
    --------
    array1 = ['C1', 'H12']
    array2 = ['H12', 'C1', 'H13', 'H14']
    returns [1, 0]

    Parameters
    ----------
    array1: numpy.ndarray
    array2: numpy.ndarray

    Returns
    -------
    np.ndarray
        The positions of the elements of array1 in array2.
    """
    if array1.size > array2.size:
        logger = setup_logger(__name__)
        logger.error(f'Condition array1.size <= array2.size is not met. '
                     f'array1 has size {array1.size} and array2 has size {array2.size}.')
        sys.exit(-1)
    idx = np.argsort(array2)
    array2_sorted = array2[idx]
    idx_sorted = np.searchsorted(array2_sorted, array1)
    array1_idx = np.take(idx, idx_sorted, mode="clip")
    pos = np.isin(array2[array1_idx], array1)
    return array1_idx[pos]


def remove_diag(M):
    """Remove the diagonal elements from a square matrix."""
    mask = ~np.eye(M.shape[0], dtype=bool)
    N = M[mask]
    N.shape = (M.shape[0], -1)
    return N


def lists_to_dict(key_, value_):
    """Convert two lists (one storing keys and the other values) to a dictionary."""
    dict_ = defaultdict(list)
    for k, v in zip(key_, value_):
        dict_[k].append(v)
    dict_ = dict(dict_)
    return dict_


def check_keys(dict_, keys_required):
    """
    Check if all keys in keys_required are present in dict_.

    Log an error and exits if a key is missing.
    """
    for key_ in keys_required:
        if key_ not in dict_:
            logger = setup_logger(__name__)
            logger.error(f'Option -{key_} is missing.')
            sys.exit(-1)


def remove_keys(dict_, keys_required):
    """Remove all keys from dict_ that are not in keys_required."""
    dict_new = {key_: value_ for key_, value_ in dict_.items() if key_ in keys_required}
    dict_.clear()
    dict_.update(dict_new)


def reorder_atom_group(array1, ref_atoms):
    """
    Reorders the elements of an AtomGroup according to a reference atom array.

    Examples
    --------
    array1 = AtomGroup containing ['C1', 'H12']
    ref_atoms = ['H12', 'C1', 'H13', 'H14']
    returns AtomGroup containing ['H12', 'C1']

    Parameters
    ----------
    array1 : MDAnalysis.core.groups.AtomGroup
        The AtomGroup to be reordered.
    ref_atoms : numpy.ndarray
        The reference atom array.

    Returns
    -------
    MDAnalysis.core.groups.AtomGroup: The reordered AtomGroup.
    """
    idx = element_idx(ref_atoms, array1.names)
    return array1[idx]
