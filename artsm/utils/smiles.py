import pandas as pd
from MDAnalysis.topology.guessers import guess_types
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from artsm.utils.bond_angle_lists import derive_bond_list
from artsm.utils.containers import idx_atoms_f

idx_to_rdkit_bond = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE,
                     4: Chem.BondType.QUADRUPLE, 5: Chem.BondType.QUINTUPLE, 6: Chem.BondType.HEXTUPLE,
                     7: Chem.BondType.AROMATIC}

rdkit_bond_to_idx = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
                     Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
                     Chem.BondType.AROMATIC: 7}


def generate_rdkit_atom(element, name=None):
    """
    Generate an RDKit atom object based on the given element symbol.

    Parameters
    ----------
    element : str
        The element symbol.
    name : str, default None
        The name of the atom.

    Returns
    -------
    rdkit.Chem.Atom
    """
    if len(element) > 1:
        element = f'{element[0]}{element[1:].lower()}'
    atom = Chem.Atom(element)
    if name is not None:
        info = Chem.AtomMonomerInfo()
        info.SetName(name)
        atom.SetMonomerInfo(info)
    return atom


def get_rdkit_bond_type(bond, table=None):
    """
    Map an integer to RDKit bond type.

    Examples
    --------
    1 -> SINGLE
    2 -> DOUBLE
    """
    if table is None:
        table = idx_to_rdkit_bond
    return table[bond]


def generate_rdkit_mol(elements, bond_list, atoms=None):
    """
    Generate RDKit molecule from an array of elements and bond list.

    Parameters
    ----------
    elements : np.ndarray
        Array of element names.

    bond_list : np.ndarray
        Array of atom bonds.
        Format: [[i, j, bond_type], ...]
        i and j are the indices of the atoms in elements.

    atoms : np.ndarray, default None
        Array of atom names.

    Returns
    -------
    mol : rdkit.Chem.rdchem.RWMol
        RDKit molecule.
    """
    # Create molecule manually
    mol = Chem.RWMol()

    # Add atoms
    for i in range(elements.size):
        if atoms is not None:
            atom = generate_rdkit_atom(elements[i], atoms[i])
        else:
            atom = generate_rdkit_atom(elements[i])
        mol.AddAtom(atom)

    # Add bonds
    for i, j, bond_type in bond_list:
        i = int(i)
        j = int(j)
        if mol.GetBondBetweenAtoms(i, j) is None:
            mol.AddBond(i, j, get_rdkit_bond_type(bond_type))

    Chem.SanitizeMol(mol)

    return mol


def canonical_atom_order(atoms, A):
    """
    Canonical atom order according to RDKit.

    RDKit molecule is generated from an array of atom names and an adjacency matrix, which specifies the connectivity
    of the atoms. Thereby, only heavy atoms are used. Afterwards, the canonical atom order is derived
    and the hydrogens are appended in the order of the heavy atoms they are connected to.
    Canonical atom order is necessary to match the atom names of molecules that are structurally the same, but have
    different atom names.

    Parameters
    ----------
    atoms : np.ndarray
        Array of atom names.
    A : np.ndarray
        Adjacency matrix of the molecule.

    Returns
    -------
    np.ndarray
        Array of atom names in canonical order.
    """
    # Sort in alphabetical order
    idx = np.argsort(atoms)
    atoms_sorted = atoms[idx]
    A_sorted = A[np.ix_(idx, idx)]
    elements = guess_types(atoms_sorted)

    # Separate heavy atoms and hydrogens
    idx = idx_atoms_f(atoms_sorted)
    atoms_f = atoms_sorted[idx]
    atoms_h = atoms_sorted[~idx]
    # adjancency matrix of heavy atoms (connected to heavy atoms)
    A_f = A_sorted[np.ix_(idx, idx)]
    # adjancency matrix of heavy atoms connected to hydrogens
    A_f_h = A_sorted[np.ix_(idx, ~idx)]
    elements_f = elements[idx]

    # Generate molecule
    # get bond list from adjacency matrix involving only heavy atoms
    bond_list = derive_bond_list(A_f)
    mol = generate_rdkit_mol(elements_f, bond_list, atoms_f)

    # Reorder heavy atoms
    rank = list(Chem.CanonicalRankAtoms(mol, includeChirality=False))
    atoms_f_ranked = atoms_f.copy()
    # reorder the arr of heavy atoms according to the canonical ranking
    # example: atoms_f = ['C1', 'C2', 'C3', 'C4']
    #          rank = [1, 3, 2, 0]
    #          atoms_f_ranked = ['C4', 'C1', 'C3', 'C2']
    atoms_f_ranked[rank] = atoms_f.copy()
    # reorder the adjacency matrix of heavy atoms (only order rows)
    A_f_h[rank] = A_f_h.copy()

    # Append H atoms in order
    idx = np.where(A_f_h > 0)[1] # only get the column indices
    # example: idx = [3, 0, 2, 1, 4]
    #         atoms_h = ['H1', 'H2', 'H3', 'H4', 'H5']
    #         atoms_h[idx] = ['H4', 'H1', 'H3', 'H2', 'H5']
    atoms_h = atoms_h[idx]
    atoms_ordered = np.concatenate((atoms_f_ranked, atoms_h))

    return atoms_ordered
