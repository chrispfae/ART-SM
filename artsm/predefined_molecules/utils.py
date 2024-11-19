import sys

import MDAnalysis as mda
import numpy as np

from artsm.predefined_molecules.classes import Water, Ion
from artsm.predefined_molecules.data import supported_water_types, supported_ion_types
from artsm.utils.other import setup_logger


def correct_predefined_molecules(aa, molecule, ion_water=None):
    """
    Modify the residue indices for snapshots containing predefined molecules like water and ion.

    One coarse grain bead represents multiple atomistic molecules. The mismatch in indices gets corrected.
    Parameters
    ----------
    aa : MDAnalysis.Universe
        Atomistic snapshot.
    molecule : str
        The residue name of the predefined molecule.
    ion_water : str
        All atoms of 'molecule' except the first one are considered water. Their resnames are replaced by this string. 
        This is necessary for instance for ions, where additionally water is present.
    Returns
    -------
    MDAnalysis.Universe
        Atomistic snapshot with corrected indices.
    """
    idx = np.where(aa.residues.resnames == molecule)[0]
    if idx.size == 0:
        return aa
    atoms_mol = aa.residues[idx[0]].atoms.names
    if ion_water is None:
        n_mols_per_mol = np.sum(atoms_mol == atoms_mol[0])
        atoms_per_mol = int(len(atoms_mol) / n_mols_per_mol)
        idx_mol = np.repeat(np.arange(n_mols_per_mol), atoms_per_mol)
    else:
        n_mols_per_mol = np.sum(atoms_mol == atoms_mol[1]) + 1
        atoms_per_mol = int((len(atoms_mol)-1) / (n_mols_per_mol-1))  # -1 necessary to exclude ion
        idx_mol_water = np.repeat(np.arange(n_mols_per_mol-1), atoms_per_mol)
        idx_mol = np.insert(idx_mol_water + 1, 0, 0)
    offset_per_mol = n_mols_per_mol - 1

    # modify residue idx
    residue_idx = []
    offset = 0
    for residue in aa.residues:
        if residue.resname == molecule:
            residue_idx.extend(idx_mol + residue.resindex + offset)
            offset += offset_per_mol
        else:
            residue_idx.extend(np.repeat(residue.resindex + offset, len(residue.atoms)))

    # modify residue names
    resnames = []
    if ion_water is None:
        mol_resnames = np.repeat(molecule, n_mols_per_mol)
    else:
        mol_resnames = np.repeat(ion_water, n_mols_per_mol)
        mol_resnames[0] = molecule
    for i, resname in enumerate(aa.residues.resnames):
        if resname == molecule:
            resnames.extend(mol_resnames)
        else:
            resnames.append(resname)

    # extract number of atoms, residues
    n_atoms = aa.atoms.n_atoms
    n_residues = len(resnames)

    aa_new = mda.Universe.empty(n_atoms, n_residues=n_residues, atom_resindex=residue_idx, trajectory=True)
    aa_new.add_TopologyAttr('name', aa.atoms.names)
    aa_new.add_TopologyAttr('resname', resnames)
    aa_new.add_TopologyAttr('resid', list(range(1, n_residues + 1)))
    aa_new.dimensions = aa.dimensions
    aa_new.atoms.positions = aa.atoms.positions
    return aa_new


def get_predefined_molecule(molecule_name='TIP3P'):
    """
    Return a PredefMol object given its name.

    Parameters
    ----------
    molecule_name : str, default 'TIP3P'
        The name of the predefined molecule to retrieve.

    Returns
    -------
    PredefMol

    Raises
    ------
    SystemExit
        If the specified molecule is not available.
    """
    if molecule_name in supported_water_types:
        return Water(**supported_water_types[molecule_name])
    elif molecule_name in supported_ion_types:
        return Ion(**supported_ion_types[molecule_name])
    else:
        logger = setup_logger(__name__)
        logger.error(f'The requested predefined molecule {molecule_name} is not available. '
                     f'However, the water types {supported_water_types.keys()} and ion types '
                     f'{supported_ion_types.keys()} are supported.')
        sys.exit(-1)


def group_water(aa, water):
    """
    Group water molecules at the end of 'aa', which eases the generation of subsequent topology files.

    Parameters
    ----------
    aa : MDAnalysis.AtomGroup
         Atomistic snapshot.
    water : str
        Water molecules to group.
    Returns
    -------
    MDAnalysis.Universe
        Atomistic snapshot with grouped water molecules.
    """
    # idx for reorder of resids
    idx_water = np.where(aa.residues.resnames == water)[0]
    idx_no_water = np.where(aa.residues.resnames != water)[0]
    idx_new = np.concatenate((idx_no_water, idx_water))
    # Reorder
    resids = aa.residues.resids
    aa_new = aa.residues[idx_new]
    aa_new.resids = resids
    return aa_new.atoms

