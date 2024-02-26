import MDAnalysis as mda
import numpy as np


def correct_water(aa, water):
    """
    Modify the residue indices for snapshots containing water.

    One coarse grain bead represents multiple atomistic water molecules. The mismatch in indices gets corrected.
    Parameters
    ----------
    aa : MDAnalysis.Universe or MDAnalysis.AtomGroup
        Atomistic snapshot.
    water : str
        Water residue name.
    Returns
    -------
    MDAnalysis.Universe
        Atomistic snapshot with corrected indices.
    """
    idx = np.where(aa.residues.resnames == water)[0]
    if idx.size == 0:
        return aa
    atoms_water = aa.residues[idx[0]].atoms.names
    n_mols_per_water = np.sum(atoms_water == atoms_water[0])
    offset_per_water = n_mols_per_water - 1
    idx_water = np.repeat(np.arange(n_mols_per_water), int(len(atoms_water) / n_mols_per_water))

    # modify residue idx
    residue_idx = []
    offset = 0
    for residue in aa.residues:
        if residue.resname == water:
            residue_idx.extend(idx_water + residue.resindex + offset)
            offset += offset_per_water
        else:
            residue_idx.extend(np.repeat(residue.resindex + offset, len(residue.atoms)))

    # modify residue names
    resnames = []
    water_resnames = np.repeat(water, n_mols_per_water)
    for i, resname in enumerate(aa.residues.resnames):
        if resname == water:
            resnames.extend(water_resnames)
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