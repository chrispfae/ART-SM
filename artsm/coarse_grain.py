import os
import sys
import warnings

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np

from artsm.utils.cli import parse_cl_coarse_graining
from artsm.utils.fileparsing import read_yaml, setup_logger
from artsm.utils.other import mda_selection
from artsm.predefined_molecules.data import supported_predefined_molecules


def read_simulation(args):
    """
    Read simulation files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI. Contains paths to simulation files.
    Returns
    -------
    MDAnalysis.Universe
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if args.x is None:
            u = mda.Universe(args.a)
        else:
            u = mda.Universe(args.a, args.x)
    return u


def extract_mapping(filenames):
    """
    Extract the mapping fields from the yaml file.

    Parameters
    ----------
        filenames (str): The path to the yaml file.

    Returns
    -------
        dict: The mapping.

    Raises
    ------
        SystemExit: If the mapping is not correctly specified for a molecule.
    """
    dict_ = {}
    for filename in filenames:
        dict_.update(read_yaml(filename))
    mapping = {}
    for mol, data in dict_.items():
        if isinstance(data, str):
            if data in supported_predefined_molecules:
                mapping[mol] = data
            else:
                logger = setup_logger(__name__)
                logger.error(f'Mapping is not specified for molecules {mol}.')
                sys.exit(-1)
        else:
            mapping[mol] = data['mapping']
    return mapping


def initialize_cg_universe(aa, mapping):
    """
    Initialize a coarse-grained universe based on the given atomistic universe and mapping.

    Not all attributes of the final cg universe are updated. Be careful when directly using this universe object.

    Parameters:
        aa (AtomisticUniverse): The atomistic universe.
        mapping (dict): A dictionary mapping residue names to bead names.

    Returns:
        cg (CoarseGrainedUniverse): The coarse-grained universe.

    """
    bead_names = {mol: list(data.keys()) for mol, data in mapping.items() if isinstance(data, dict)}
    n_beads = {mol: len(data) for mol, data in bead_names.items() if data}

    cg_beads = []
    atomistic_atoms_idx = []  # Selecting the correct atoms to get the number of residues right.
    offset = 0
    n_residues = 0
    for residue in aa.residues:
        residue_name = residue.resname
        if isinstance(mapping[residue_name], str) and mapping[residue_name] in supported_predefined_molecules:
            offset += residue.atoms.n_atoms
            continue
        cg_beads.extend(bead_names[residue_name])
        atomistic_atoms_idx.extend(np.arange(n_beads[residue_name]) + offset)
        n_residues += 1
        offset += residue.atoms.n_atoms

    cg = aa.atoms[atomistic_atoms_idx]
    cg = mda.Merge(cg)
    cg.del_TopologyAttr('name')
    cg.add_TopologyAttr('name', cg_beads)
    cg.del_TopologyAttr('resid')
    cg.add_TopologyAttr('resid', np.arange(n_residues) + 1)

    return cg


def coarse_grain(aa, cg, mapping):
    if aa.trajectory.n_frames == 1:
        coordinates = np.zeros((cg.atoms.n_atoms, 3))
        count = 0
        for id_, residue in enumerate(aa.residues):
            if isinstance(mapping[residue.resname], str) and mapping[residue.resname] in supported_predefined_molecules:
                continue
            for atoms_names in mapping[residue.resname].values():
                selection = f'resid {id_ + 1} and ({mda_selection(atoms_names)})'
                coordinates_bead = aa.select_atoms(selection).center_of_mass()
                coordinates[count, :] = coordinates_bead
                count += 1
        cg.atoms.positions = coordinates
        cg.dimensions = aa.dimensions
    else:
        coordinates = np.zeros((aa.trajectory.n_frames, cg.atoms.n_atoms, 3))
        count = 0
        for id_, residue in enumerate(aa.residues):
            if isinstance(mapping[residue.resname], str) and mapping[residue.resname] in supported_predefined_molecules:
                continue
            for atoms_names in mapping[residue.resname].values():
                selection = f'resid {id_ + 1} and ({mda_selection(atoms_names)})'
                coordinates_bead = AnalysisFromFunction(lambda atoms: atoms.center_of_mass(),
                                                        aa.select_atoms(selection)).run().results['timeseries']
                coordinates[:, count, :] = coordinates_bead
                count += 1
        cg.load_new(coordinates, format=MemoryReader)
        for ts in range(aa.trajectory.n_frames):
            aa.trajectory[ts]
            cg.trajectory[ts]
            cg.dimensions = aa.dimensions


def output_pdb_xtc(cg, outfiles, traj):
    """
    Write the PDB and XTC files based on the given coarse-grained (cg) object and path.

    If the trajectory in the coarse-grained object has more than one frame then the 
    XTC file is written as well.

    Parameters:
    - cg: The coarse-grained object
    - path: The path to save the PDB and XTC files

    Returns:
    None
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if len(outfiles) == 2 and traj is None:
            logger = setup_logger(__name__)
            logger.warning('It seems you have requested an output trajectory without providing an input trajectory. '
                           'I can thus only provide an output snapshot.')
            cg.atoms.write(outfiles[0])
        elif len(outfiles) == 1:
            cg.atoms.write(outfiles[0])
        elif len(outfiles) == 2 and traj is not None and cg.trajectory.n_frames > 1:
            cg.atoms.write(outfiles[0])
            with mda.Writer(outfiles[1], cg.atoms.n_atoms) as W:
                for _ in cg.trajectory:
                    W.write(cg.atoms)
        else:
            logger = setup_logger(__name__)
            logger.error('Writing to output file failed. Maybe the following went wrong:\n'
                         'More than 2 arguments were provided for -o.\n'
                         'The input trajectory contains only 1 frame.\n'
                         'Something unexpected occurred during the coarse-graining process.')
            sys.exit(-1)


def main():
    args = parse_cl_coarse_graining(sys.argv[1:])
    mapping = extract_mapping(args.t)
    aa = read_simulation(args)

    cg = initialize_cg_universe(aa, mapping)
    coarse_grain(aa, cg, mapping)
    output_pdb_xtc(cg, args.o, args.x)


if __name__ == '__main__':
    main()
