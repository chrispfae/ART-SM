import sys

import numpy as np

from artsm.utils.other import setup_logger


def derive_bond_list(A):
    """
    Derive a bond list between atoms in a molecule from the adjacency matrix A.

    Extracts the lower triangular part of the adjacency matrix A and returns the indices of the non-zero elements.
    For each index the bond type is extracted from the adjacency matrix. The final bond list arr is created by
    stacking the indices and bond types.

    Example:
    A = np.arr([[0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0],
                  [0, 1, 0, 1, 1],
                  [1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]])

    L = np.arr([[0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]])

    idx = np.arr([[1, 0],
                    [2, 1],
                    [3, 0],
                    [3, 2],
                    [4, 2]])

    bond_type = np.arr([1, 1, 1, 1, 1])

    bond_list = np.arr([[1, 0, 1],
                            [2, 1, 1],
                            [3, 0, 1],
                            [3, 2, 1],
                            [4, 2, 1]])
                            
    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the molecule. Value indicates the bond type.

    Returns
    -------
    np.ndarray
        Bond list.
    """
    L = np.tril(A)
    idx = np.where(L >= 1)
    bond_type = np.ones(idx[0].size, dtype=np.int_)
    # populate bond_type using the bond types obtained from the adjacency matrix
    for i in range(idx[0].size):
        bond_type[i] = L[idx[0][i], idx[1][i]]
    bond_list = np.vstack((idx[0], idx[1], bond_type)).T
    return bond_list


def derive_angle_list(A):
    """
    Derives a list of angles between atoms in a molecule from the adjacency matrix A.

    The angle list is derived based on the bond list, which gets calculated from the given adjacency matrix.
    Each bond in the bond list is subsequently expanded to a third atom.
    Finally, duplicate angles are removed from the angle list using the remove_duplicate_angles function.
    Each entry of the angle list has the form
    [idx_atom1, idx_atom2, idx_atom3, bond_type_atom1_atom2, bond_type_atom2_atom3]

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the molecule. Value indicates the bond type.

    Returns
    -------
    numpy.ndarray
        Angle list.
    """
    bond_list = derive_bond_list(A)
    angles = None
    for atom1, atom2, bond_type in bond_list:
        angles_new = get_angles(atom1, atom2, bond_type, bond_list)
        if angles is None:
            angles = angles_new
        else:
            angles = np.row_stack((angles, angles_new))

    angles = remove_duplicate_angles(angles)
    return angles


def remove_duplicate_angles(angle_list):
    """
    Remove duplicate angles from the angle list.

    Each entry of the angle list has the form
    [idx_atom1, idx_atom2, idx_atom3, bond_type_atom1_atom2, bond_type_atom2_atom3].
    Some entries may describe the same angle. For example, the angle between
    atom 1, atom 2, and atom 3 is the same as the angle between atom 3, atom 2,
    and atom 1. This function removes the duplicate angles.

    Parameters
    ----------
    angle_list : numpy.ndarray
        Angle list.

    Returns
    -------
    numpy.ndarray
    Angle list without duplicate angles.
    """
    angle_list_unique = np.unique(angle_list, axis=0)
    idx = []
    for i in range(angle_list_unique.shape[0]):
        for j in range(i + 1, angle_list_unique.shape[0]):
            arr1 = angle_list_unique[i]
            arr2 = angle_list_unique[j]
            arr2_ordered = np.array([arr2[2], arr2[1], arr2[0], arr2[3], arr2[4]])
            if (arr1 == arr2_ordered).all():
                idx.append(j)
    return np.delete(angle_list_unique, idx, axis=0)


def get_angles(atom1, atom2, bond_type, bond_list):
    """
    Determines all possible angles from a given bond list that contain atom1 and atom2.

    Parameters
    ----------
    atom1 : int
        Index of the first atom.
    atom2 : int
        Index of the second atom.
    bond_type : int
        Type of the bond between atom1 and atom2.
    bond_list : numpy.ndarray
        Bond list.

    Returns
    -------
    numpy.ndarray
        Angle list. Each angle has the form
        [idx_atom1, idx_atom2, idx_atom3, bond_type_atom1_atom2, bond_type_atom2_atom3].
    """
    angles1 = _get_angle(atom1, atom2, bond_type, bond_list, 0)
    angles2 = _get_angle(atom1, atom2, bond_type, bond_list, 1)
    angles3 = _get_angle(atom2, atom1, bond_type, bond_list, 0)
    angles4 = _get_angle(atom2, atom1, bond_type, bond_list, 1)
    angles = np.row_stack((angles1, angles2, angles3, angles4))
    return angles


def _get_angle(atom1, atom2, bond_type, bond_list, i):
    """
    Determines angles from a given bond list that contain atom1 and atom2.

    First, atoms are determined that are bonded to atom1, but not atom2. Subsequently, the bond types are derived.
    Note that the bond_list might not contain duplicate entries, i.e. bond atom1-atom2, but not atom2-atom1. In this
    case the function has to be called with i=0 and i=1. Thus, not necessarily all possible angles are returned.

    Parameters:
    atom1 : int
        Index of the first atom.
    atom2 : int
        Index of the second atom.
    bond_type : int
        Type of the bond between atom1 and atom2.
    bond_list : numpy.ndarray
        Bond list.
    i : int
        Bond list column. Searches only this column for occurrences of atom1. Either 0 or 1.

    Returns
    -------
    numpy.ndarray
        Angles that contain atom1 and atom2.
    """
    if i == 0:
        j = 1
    elif i == 1:
        j = 0
    else:
        logger = setup_logger(__name__)
        logger.error(f'Parameter i has to be either 0 or 1. The value {i} was given.')
        sys.exit(-1)
    idx = np.where(bond_list[:, i] == atom1)[0]
    bonded = bond_list[idx, j]
    idx_nj = np.where(bonded != atom2)[0]
    bonded = bonded[idx_nj]
    bonded_types = bond_list[idx, 2]
    bonded_types = bonded_types[idx_nj]
    return np.column_stack((bonded, np.full_like(bonded, atom1), np.full_like(bonded, atom2),
                            bonded_types, np.full_like(bonded, bond_type)))
