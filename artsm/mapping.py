import os
import sys
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix

from artsm.utils.cli import parse_cl_mapping
from artsm.utils.containers import extract_atoms_f, lists_to_dict, remove_diag, remove_duplicates
from artsm.utils.fileparsing import write_yaml
from artsm.utils.other import setup_logger
from artsm.utils.smiles import rdkit_bond_to_idx

import MDAnalysis as mda


def get_names(mol):
    """
    Returns the names of the atoms in a molecule.
    """
    atoms = mol.GetAtoms()
    names = np.empty(len(atoms), dtype=object)
    for i, atom in enumerate(atoms):
        names[i] = atom.GetMonomerInfo().GetName().strip()
    return names


def combine_mol_smiles(mols, smiles):
    """
    Combines a list of molecules with a given SMILES string.

    Args:
        mols (list): A list of RDKit molecule objects.
        smiles (str): The SMILES string representing the template molecule.

    Returns:
        mol_new (RDKit molecule): The combined molecule with bond orders assigned from the template.

    Raises:
        ValueError: If the number of atoms in the template molecule does not match the
                    number of atoms in the input molecules.

    """
    # Check that number of atoms match
    names = get_names(mols)
    names_f = extract_atoms_f(names)
    n_mol = names_f.size

    template = Chem.MolFromSmiles(smiles)
    n_template = template.GetNumAtoms()

    if n_mol != n_template:
        logger = setup_logger(__name__)
        logger.error(f'The smiles \'{smiles}\' does not match the pdb structure.')
        sys.exit(-1)

    # Create new molecule
    try:
        mol_new = AllChem.AssignBondOrdersFromTemplate(template, mols)
    except ValueError:
        logger = setup_logger(__name__)
        logger.error(f'The smiles \'{smiles}\' does not match the pdb structure.')
        sys.exit(-1)

    # Get charge info
    charges = {}
    for atom in mol_new.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge != 0:
            atom_name = atom.GetMonomerInfo().GetName().strip()
            charges[atom_name] = charge
    return mol_new, charges


def derive_adj_atoms(mol):
    # Initialize adjacency atoms
    names = get_names(mol)
    adj_atoms = {}
    for name_ in names:
        adj_atoms[name_] = []

    # Determine bonds and bond type. Add to adj_atoms.
    bonds = mol.GetBonds()
    for bond in bonds:
        atom1 = bond.GetBeginAtom().GetMonomerInfo().GetName().strip()
        atom2 = bond.GetEndAtom().GetMonomerInfo().GetName().strip()
        bond_type = rdkit_bond_to_idx[bond.GetBondType()]
        if bond_type > 1:
            atom1_mod = f'{atom1}-{bond_type}'
            atom2_mod = f'{atom2}-{bond_type}'
        else:
            atom1_mod = atom1
            atom2_mod = atom2

        adj_atoms[atom1].append(atom2_mod)
        adj_atoms[atom2].append(atom1_mod)

    return adj_atoms


def _sanity_check(idx):
    """
    Perform a sanity check on the given index arr.

    Parameters:
    idx (numpy.ndarray): The index arr to be checked.

    Raises:
    SystemExit: If any atom in the index arr is assigned twice.

    """
    idx_unique = remove_duplicates(idx)
    if idx.size != idx_unique.size:
        logger = setup_logger(__name__)
        logger.error("Atomistic atom got assigned twice.")
        sys.exit(-1)


def _derive_bonded_atoms(atom):
    """
    Derives the indices of the atoms bonded to the given atom.

    Parameters:
        atom (rdkit.Chem.Atom): The atom for which to derive the bonded atoms.

    Returns:
        numpy.ndarray: An arr containing the indices of the bonded atoms.
    """
    atom_idx = atom.GetIdx()
    bonds = atom.GetBonds()
    bonded_idx = np.ones(len(bonds), dtype=np.int_)
    for i, bond in enumerate(bonds):
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 == atom_idx:
            bonded_idx[i] = idx2
        else:
            bonded_idx[i] = idx1
    return bonded_idx


def _assign_missing_atoms(atoms_cg, atoms_aa, mol_aa, names_cg):
    assigned_all = True
    n = mol_aa.GetNumAtoms()
    unassigned_aa = np.array([i for i in np.arange(n) if i not in atoms_aa])
    assigned_cg = []
    assigned_aa = []
    for atom_idx in unassigned_aa:
        atom = mol_aa.GetAtomWithIdx(int(atom_idx))
        bonded_atoms = _derive_bonded_atoms(atom)
        if bonded_atoms.size == 0:
            logger = setup_logger(__name__)
            logger.error(f'I have not found a bond for atom {atom.GetMonomerInfo().GetName()}.')
            sys.exit(-1)
        else:
            assigned_bonded = np.isin(atoms_aa, bonded_atoms)
            if np.sum(assigned_bonded) == 0:
                assigned_all = False
            else:
                candidates_cg = set(atoms_cg[assigned_bonded])
                assigned_bead = candidates_cg.pop()
                if len(candidates_cg) > 1:
                    logger = setup_logger(__name__)
                    logger.warning(f'Atom {atom.GetMonomerInfo().GetName()} can not be unambiguously assigned. '
                                   f'I have chosen bead {names_cg[assigned_bead]}')
                assigned_cg.append(assigned_bead)
                assigned_aa.append(atom_idx)
    assigned_cg = np.concatenate((atoms_cg, assigned_cg))
    assigned_aa = np.concatenate((atoms_aa, assigned_aa))
    return assigned_cg, assigned_aa, assigned_all


def derive_mapping(coords_cg, bead_names, mol_aa):
    """
    For each bead:
    1. Compute the smallest distance d to any other bead.
    2. Assign atom if its distance to the current bead is < d/2
    For all remaining atoms:
    1. Check if connected atoms are assigned to a bead B.
       Assign the current atom to B.
       If ambiguous the assignment is randomly chosen, but a warning is printed.
    2. Repeat 1. until all atoms are assigned
    Parameters
    ----------
    coords_cg: np.ndarray
        Coordinates of coarse-grained structure
    mol_aa: rdkit.Chem.mol
        atomistic structure
    Returns
    -------
    dict:
        A dictionary with coarse-grained beads as keys and a list of atoms as values.
    """
    # Distances to neighbor atoms
    D = distance_matrix(coords_cg, coords_cg)
    D = remove_diag(D)
    if D.size == 0:
        mapping = {bead_names[0]: list(get_names(mol_aa))}
        return mapping
    r = np.min(D, axis=1) / 2

    # Assign atoms to beads
    coords_aa = mol_aa.GetConformer(0).GetPositions()
    D = distance_matrix(coords_cg, coords_aa)
    assigned_cg, assigned_aa = np.where((D < r[:, None]))
    _sanity_check(assigned_aa)

    # Assign remaining atoms
    if assigned_aa.size != mol_aa.GetNumAtoms():
        assigned_all = False
        while not assigned_all:
            assigned_cg, assigned_aa, assigned_all = _assign_missing_atoms(assigned_cg, assigned_aa, mol_aa, bead_names)

    # Convert indices to names
    names_aa = get_names(mol_aa)
    mapping = lists_to_dict(bead_names[assigned_cg], names_aa[assigned_aa])

    return mapping


def get_mol_name(mol):
    """
    Returns the name of the molecule.

    This is the residue name of the first atom in the molecule.
    """
    return mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName()


def get_res_idx(mol):
    """
    Returns the residue index of the molecule.

    This is the residue number of the first atom in the molecule.
    """
    return mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueNumber()


def unique_mols_aa(snapshot):
    """
    Returns a list of unique molecule names and their corresponding molecules from a given snapshot.

    We loop over all molecules in the snapshot and check if the molecule name is already in the list of seen molecules.
    If not, we add the molecule name to the list of seen molecules and append the molecule to the list of relevant molecules.

    Parameters:
    snapshot (Chem.Mol): The snapshot containing molecules.

    Returns:
    tuple: A tuple containing two lists - the first list contains unique molecule names, and the second list contains the corresponding unique molecules.
    """

    mols = Chem.GetMolFrags(snapshot, asMols=True)
    seen = []
    relevant_mols = []

    for i, mol in enumerate(mols):
        name = get_mol_name(mol)
        if name not in seen:
            seen.append(name)
            relevant_mols.append(mol)

    return seen, relevant_mols


def unique_mols_cg(snapshot):
    res_names = snapshot.residues.resnames
    seen = []
    coords = []
    bead_names = []
    for i, name in enumerate(res_names):
        if name not in seen:
            seen.append(name)
            resid = snapshot.select_atoms(f'resid {i + 1}')
            coords.append(resid.positions)
            bead_names.append(resid.names)
    return seen, coords, bead_names


def get_atom_order(mol):
    """
    Returns the order of the atoms in a molecule. The order is determined by the order of the atoms provided
    by the molecule.GetAtoms() function. 
    """
    atom_order = []
    for atom in mol.GetAtoms():
        name = atom.GetMonomerInfo().GetName()
        atom_order.append(name.strip())
    return atom_order


def main():
    args = parse_cl_mapping(sys.argv[1:])
    mol_smiles = {}
    for mol, smiles in args.s:
        mol_smiles[mol] = smiles

    snapshot_aa = Chem.MolFromPDBFile(args.a, removeHs=False)
    mol_names_aa, mols_raw_aa = unique_mols_aa(snapshot_aa)
    atom_order = [get_atom_order(mol) for mol in mols_raw_aa]
    smiles = [mol_smiles[name] for name in mol_names_aa]

    mols_aa = []
    charges = []
    for mol, s in zip(mols_raw_aa, smiles):
        mol, charge = combine_mol_smiles(mol, s)
        mols_aa.append(mol)
        charges.append(charge)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        snapshot_cg = mda.Universe(args.m)

    mol_names_cg, coords_cg, bead_names = unique_mols_cg(snapshot_cg)

    if not mol_names_aa == mol_names_cg:
        logger = setup_logger(__name__)
        logger.error(f'Residue names in coarse-grained {mol_names_cg} and '
                     f'atomistic {mol_names_aa} snapshot do not match.')
        sys.exit(-1)

    adj_atoms = [derive_adj_atoms(mol) for mol in mols_aa]
    mapping = [derive_mapping(coords, names, mol_aa) for coords, names, mol_aa in zip(coords_cg, bead_names, mols_aa)]

    mapping_dict = {}
    for i in range(len(mol_names_aa)):
        # Account for water
        if smiles[i] == 'O':
            mapping_dict[mol_names_aa[i]] = f'{args.w}'
        else:
            mapping_dict[mol_names_aa[i]] = {'smiles': smiles[i], 'adj_atoms': adj_atoms[i], 'mapping': mapping[i],
                                             'charges': charges[i], 'atom_order': atom_order[i]}
    write_yaml(args.o, mapping_dict)


if __name__ == '__main__':
    main()
