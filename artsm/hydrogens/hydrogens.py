import biotite.structure
import hydride
import hydride.cli
import numpy as np

from artsm.utils.containers import extract_atoms_f, extract_atoms_h, idx_atoms_f


def _parse_to_biotite(molecule, coords):
    """
    Convert a Molecule object and to a Biotite AtomArray object.

    Parameters
    ----------
    molecule : Molecule
        Artsm molecule object to be converted to Biotite AtomArray.
    coords : numpy.ndarray
        The coordinates of the atoms in the molecule.

    Returns
    -------
    Biotite AtomArray
        The converted Biotite AtomArray object.
    """

    biotite_mol = biotite.structure.AtomArray(molecule.n_atoms_f)
    biotite_mol.add_annotation("charge", int)
    biotite_mol.atom_name = molecule.atoms_f.copy()
    biotite_mol.element = molecule.elements_f.copy()
    biotite_mol.coord = coords.copy()
    biotite_mol.bonds = biotite.structure.BondList(molecule.n_atoms_f, molecule.bond_list_f.copy())

    return biotite_mol


def _correct_h(mol_biotite, atom, h):
    """
    Correct the hydrogens of one atom in a molecule by removing existing and subsequently adding correct hydrogens.

    Currently not used since hydrogen atoms are not corrected for single atoms for the whole molecule.

    Parameters
    ----------
    mol_biotite : Biotite AtomArray
        Molecule object
    atom : str
        The name of the atom whose hydrogens are corrected.
    h : list of str
        The names of hydrogen atoms to be removed.
    """
    # Remove bad hydrogens
    idx_h = np.isin(mol_biotite.atom_name, h)
    mol_temp = mol_biotite[np.invert(idx_h)]

    # Add correct hydrogens
    idx_atom = mol_temp.atom_name == atom
    mol_temp, mask = hydride.add_hydrogen(mol_temp, mask=idx_atom)

    # Update coords of original molecule
    coords_h = mol_temp.coord[np.invert(mask)]
    mol_biotite.coord[idx_h] = coords_h


def correct_hydrogens_per_atom(molecule, coords):
    """
    Corrects the number and positions of hydrogen atoms in a molecule by subsequently correcting each atom individually.

    Currently not used.

    Parameters
    ----------
        molecule : Molecule
        coords : numpy.ndarray
            The coordinates of the atoms in the molecule.

    Returns
    -------
    numpy.ndarray
        The corrected coordinates of all atoms in the molecule.
    """
    # Atoms that require hydrogen correction
    atoms_correction = np.empty((len(molecule.fr_pairs) * 2), dtype=object)
    for i, fr_pair in enumerate(molecule.fr_pairs.values()):
        atoms_correction[i * 2:i * 2 + 2] = fr_pair.con.atoms[1:3]

    # Corresponding bonded hydrogen atoms
    A_f_h = molecule.A.loc[molecule.atoms_f][molecule.atoms_h]
    h_bonded = np.empty((atoms_correction.size, 2), dtype=object)
    for i, atom in enumerate(atoms_correction):
        h = A_f_h.columns[A_f_h.loc[atom] == 1]
        if h.size == 0:
            continue
        else:
            h_bonded[i] = A_f_h.columns[A_f_h.loc[atom] == 1]

    # Setup biotite molecule
    if molecule.bond_list is None:
        molecule.derive_bond_list()
    mol_biotite = _parse_to_biotite(molecule, coords)

    # Correct one atom at a time
    for atom, h in zip(atoms_correction, h_bonded):
        _correct_h(mol_biotite, atom, h)

    return mol_biotite.coord


def correct_hydrogens(molecule, coords):
    """
    Adds hydrogen atoms to the given molecule and returns the coordinates of the modified molecule.

    The molecule is first converted to a Biotite AtomArray object and afterwards the hydrogens are added using the
    hydride package.

    Parameters
    ----------
    molecule : Molecule
    coords : numpy.ndarray
        The coordinates of the atoms in the molecule.

    Returns
    -------
    - The coordinates of the modified molecule with added hydrogen atoms.
    """
    # Parse to biotite molecule and add hydrogen atoms
    mol_biotite = _parse_to_biotite(molecule, coords)
    mol_biotite_h = hydride.add_hydrogen(mol_biotite)[0]

    return mol_biotite_h.coord
