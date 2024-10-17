import os
from scipy.spatial.distance import pdist, squareform
import sys
import warnings

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np

from artsm.utils.cli import parse_cl_generate_posre
from artsm.utils.containers import idx_atoms_f
from artsm.utils.fileparsing import read_yaml, setup_logger
from artsm.utils.other import mda_selection
from artsm.water.data import supported_water_models
from artsm.water.utils import correct_water


def extract_mapping(filenames):
    dict_ = {}
    for filename in filenames:
        dict_.update(read_yaml(filename))

    mapping = {}
    water = []
    for mol, data in dict_.items():
        if isinstance(data, str) and data in supported_water_models:
            mapping[mol] = {atom: 'water' for atom in supported_water_models[data]['atoms']}
            water.append(mol)
        elif isinstance(data, dict) and 'mapping' in data:
            swapped_dict = {value: key for key, values in data['mapping'].items() for value in values}
            mapping[mol] = swapped_dict
        else:
            logger = setup_logger(__name__)
            logger.error(f'Mapping is not specified for molecules {mol}.')
            sys.exit(-1)
    return mapping, water


def derive_atom_order(filenames, cg):
    dict_ = {}
    for filename in filenames:
        dict_.update(read_yaml(filename))

    atom_order = {}
    for mol, data in dict_.items():
        if isinstance(data, str) and data in supported_water_models:
            atom_order[mol] = supported_water_models[data]['atoms']
        elif isinstance(data, dict) and 'atom_order' in data:
            atom_order[mol] = data['atom_order']
        else:
            for residue in cg.residues:
                if residue.resname in atom_order:
                    continue
                else:
                    cg_atom_order = residue.atoms.names
                    aa_atom_order = []
                    for bead_name in cg_atom_order:
                        aa_atom_order.extend(data['mapping'][bead_name])
                    atom_order[mol] = aa_atom_order
    return atom_order


def _atomistic_universe(cg, atom_order):
    residue_names = cg.residues.resnames
    n_residues = residue_names.size
    atoms = np.array([atom_order[mol] for mol in residue_names], dtype=object)
    residue_idx = np.array([count for count, sublist in enumerate(atoms) for _ in sublist])
    atoms = np.concatenate(atoms)
    n_atoms = atoms.size
    u = mda.Universe.empty(n_atoms, n_residues=n_residues, atom_resindex=residue_idx, trajectory=True)
    u.add_TopologyAttr('name', atoms)
    u.add_TopologyAttr('resname', residue_names)
    u.add_TopologyAttr('resid', list(range(1, n_residues + 1)))
    u.dimensions = cg.dimensions
    return u


def posre(aa, cg, mapping, restrain_water):
    logger = setup_logger(__name__)
    for i, atom in enumerate(aa.atoms):
        if i % 10_000 == 0:
            logger.info(f'Processed {i} atoms')
        cg_bead = mapping[atom.resname][atom.name]
        if cg_bead == 'water':
            if restrain_water:
                selection = f'resid {atom.resid}'
            else:
                continue
        else:
            selection = f'resid {atom.resid} and name {cg_bead}'
        atom.position = cg.select_atoms(selection)[0].position


def output_gro_xtc(aa_posre, filename):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        aa_posre.atoms.write(filename)


def bead_distances(coords):
    coords_unique = np.unique(coords, axis=0)
    D = squareform(pdist(coords_unique))
    np.fill_diagonal(D, np.inf)
    minima = np.min(D, axis=1)
    return np.mean(minima)


def output_posre_itp(aa_posre, radius, filename, water, restrain_water):
    residues_seen = []
    for residue in aa_posre.residues:
        if residue.resname in residues_seen:
            continue
        if residue.resname in water and not restrain_water:
            continue
        else:
            atoms = residue.atoms.names
            idx_heavy = np.where(idx_atoms_f(atoms))[0]
            output_file_name = os.path.join(os.path.dirname(filename),
                                            f'{residue.resname}_{os.path.basename(filename)}')
            output_file = open(output_file_name, 'w')
            output_file.write('[ position_restraints ]\n')
            output_file.write(';ai    func    g    r    k\n')
            for i in idx_heavy:
                output_file.write(f'{i + 1}    2    1    {radius}    1000\n')
            output_file.close()


def main():
    args = parse_cl_generate_posre(sys.argv[1:])
    mapping, water = extract_mapping(args.t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cg = mda.Universe(args.c)
    atom_order = derive_atom_order(args.t, cg)
    aa = _atomistic_universe(cg, atom_order)
    box_dims = aa.dimensions
    posre(aa, cg, mapping, args.restrain_water)
    if args.restrain_water:
        for water_mol in water:
            aa = correct_water(aa, water_mol)
    elif water:
        selection = f'not ({mda_selection(water, "resname")})'
        aa = mda.Merge(aa.select_atoms(selection))
        aa.dimensions = box_dims

    if args.r is not None:
        output_gro_xtc(aa, args.r)
    
    if args.i is not None:
        mean_bead_dist = bead_distances(cg.atoms.positions)
        recommended_radius = round(mean_bead_dist / 20., 2)
        logger = setup_logger(__name__)
        logger.info(f'Recommended value for parameter R in flat bottom position restraint: {recommended_radius} nm.')
        logger.info(f'I will use this value for generating the position restraint itp file')
        output_posre_itp(aa, recommended_radius, args.i, water, args.restrain_water)


if __name__ == '__main__':
    main()

