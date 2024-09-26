import sys
import warnings

import MDAnalysis as mda
import numpy as np

import artsm.water.model
from artsm.hydrogens.hydrogens import correct_hydrogens
from artsm.water.data import sphere_radius
from artsm.optimization.optimization import optimize_molecule, optimize_one_bead_mol
from artsm.resolution_transformation.backmap import rotate_random
from artsm.topology.molecule import Molecule
from artsm.utils.angles import calc_angle
from artsm.utils.bond_angle_lists import derive_bond_list, derive_angle_list
from artsm.utils.containers import type_random_value, element_idx, idx_atoms_f, reorder_atom_group
from artsm.utils.fileparsing import join_path, read_yaml
from artsm.utils.other import setup_logger, center_of_mass, mda_selection
from artsm.utils.clashing_atoms import clashing_atoms
from artsm.utils.smiles import canonical_atom_order
from artsm.water.model import get_water_model, Water
from artsm.water.utils import correct_water


def _check_config(config, required_keys):
    """
    Check the content of a simulation config file.

    Required keys have to be in the config file as well as the keys 'mapping', 'smiles', and 'adj_atoms'
    in config['molecules'].

    Parameters
    ----------
    config : dict
        Simulation config file.
    required_keys : list of str
        Required keys that have to be in the config file.

    Raises
    ------
    SystemExit
        If specific keys are missing in the simulation config file.
    """
    if 'path' in config:
        simulation_path = config['path']
    else:
        simulation_path = 'config file.'
    # Check first level of dictionary
    for value in required_keys:
        if value not in config:
            logger = setup_logger(__name__)
            logger.error(f'Required key "{value}" not given in {simulation_path}')
            sys.exit(-1)
    for value in config:
        if value not in required_keys:
            logger = setup_logger(__name__)
            logger.warning(f'I do not know the key "{value}" given in {simulation_path}. Key will be ignored.')

    # Check molecule specific configuration
    required_keys = ['mapping', 'smiles', 'adj_atoms']
    for molecule_name, molecule_config in config['molecules'].items():
        for value in required_keys:
            if value not in molecule_config:
                logger = setup_logger(__name__)
                logger.error(f'Required key "{value}" not given in {simulation_path} for molecule {molecule_name}.')
                sys.exit(-1)
        for value in molecule_config:
            if value not in required_keys:
                logger = setup_logger(__name__)
                logger.warning(f'I do not know the key "{value}" given in {simulation_path} for molecule '
                               f'{molecule_name}. Key will be ignored.')


def parse_snapshot(snapshot_config):
    """
    Parse a simulation config file and return an Simulation object.

    Parameters
    ----------
    snapshot_config : dict
        Simulation config file.

    Returns
    -------
    Simulation
    """
    _check_config(snapshot_config, ['molecules', 'path'])
    molecules = snapshot_config['molecules']
    return Simulation(molecules)


def _extract_timestep(traj):
    """
    Extract the time step from a simulation trajectory.

    Parameters
    ----------
    traj : MDAnalysis Trajectory, e.g. XTCReader
        Simulation trajectory.

    Returns
    -------
    float
        Trajectory time step.

    Raises
    ------
    SystemExit
        If the trajectory contains different time steps or the time is smaller equal zero.
    """
    timesteps = np.zeros(traj.n_frames)
    for i, ts in enumerate(traj):
        timesteps[i] = ts.dt
    timesteps_unique, counts = np.unique(timesteps, return_counts=True)
    if timesteps_unique.size > 1:
        logger = setup_logger(__name__)
        logger.error(f'I found different time steps in the trajectory, which can not be handled:')
        for timestep, count in zip(timesteps_unique, counts):
            logger = setup_logger(__name__)
            logger.error(f'time step: {timestep}   number of occurrences: {count}')
        sys.exit(-1)
    timestep = timesteps_unique[0]
    if timestep <= 0:
        logger = setup_logger(__name__)
        logger.error(f'The time step of {timestep} of the provided trajectory seems to be invalid.')
        sys.exit(-1)
    return timestep


def _parse_atom(atom):
    """
    Parse atom name from a simulation config file and return the name and the bond type.

    Example: 'C1-2' -> 'C1', 2

    Parameters
    ----------
    atom : str
        Atom name. Example: 'C1-2'

    Returns
    -------
    atom_name : str
        Atom name without bond type. Example: 'C1'
    bond_type : int
        Bond type. Example: 2
    """
    atom_parsed = atom.split('-')
    if len(atom_parsed) > 2:
        logger = setup_logger(__name__)
        logger.error('Atom name has too many \'-\' symbols. Please provide either pure atom name (e.g. \'C1\') '
                     'or with bond type (e.g. \'C1-2\' for double bond).')
        sys.exit(-1)
    elif len(atom_parsed) == 1:
        atom_parsed.append(1)
    return atom_parsed[0], atom_parsed[1]


def _parse_atoms(atoms):
    """
    Parse atom names from a simulation config file and return the names and the bond types.

    Example: ['C1-2', 'C2-1', 'O1-1'] -> ['C1', 'C2', 'O1'], [2, 1, 1]

    Parameters
    ----------
    atoms : list
        List of atoms of a molecule. Each atom is provided as a string.
        Example for a molecule with 3 atoms: ['C1', 'C2', 'O1']

    Returns
    -------
    atoms_parsed : np.ndarray
        Atoms of the molecule.
    bond_types : np.ndarray
        Bond types of the molecule.
    """
    atoms_parsed = np.empty(len(atoms), dtype=object)
    bond_types = np.ones(len(atoms), dtype=np.int8)
    for i, atom in enumerate(atoms):
        atoms_parsed[i], bond_types[i] = _parse_atom(atom)
    return atoms_parsed, bond_types


def _parse_adj_atoms(adj_atoms, charges=None):
    """
    Derive adjacency matrix and respective atoms from config dictionary.

    Parameters
    ----------
    adj_atoms : dict
        Contains the information of field 'adj_atoms' of a simulation config file.
    charges : dict
        Contains the information of field 'charges' of a simulation config file.

    Returns
    -------
    atoms : np.ndarray
        Atoms of the molecule.
    A : np.ndarray
        Adjacency matrix of the molecule
    atoms and A are connected -> The order matters.

    Both the atoms and the adjacency matrix are ordered according to the canonical atom order.
    """
    atoms = np.array(list(adj_atoms.keys()))
    # this adj matrix will contain the bond types for all atoms
    A = np.zeros((atoms.size, atoms.size), dtype=np.uint8)

    for atom1, atoms_connected in adj_atoms.items():
        atoms_connected, bond_types = _parse_atoms(atoms_connected)
        idx = element_idx(atoms_connected, atoms)
        # set bond types at row: atom1, column: atoms_connected
        A[(atom1 == atoms), idx] = bond_types

    # Canonical ordering of atoms
    atoms_ordered = canonical_atom_order(atoms, A, charges)
    idx = element_idx(atoms_ordered, atoms)
    # Reorder adjacency matrix
    A_ordered = A[idx][:, idx]

    return atoms_ordered, A_ordered


def _parse_topology(molecules, rng):
    """
    Parse information of individual molecules (dict) into a class Molecule instance.

    Parameters
    ----------
    molecules : dict
        Dictionary of type Molecule or dictionary with molecule information.
        Use information of smiles, adj_atoms, mapping, atom_order to create a Molecule instance.
        If atom_order is not given, canonical atom order is used.
    rng : np.random.default_rng()
        Default random number generator of numpy.

    Returns
    -------
    molecules : dict
        Dictionary of type Molecule.
    """
    if type_random_value(molecules, rng) != Molecule:
        for molecule_name, molecule in molecules.items():
            if isinstance(molecule, str):
                molecules[molecule_name] = get_water_model(molecule)
            else:
                smiles = molecule['smiles']
                adj_atoms = molecule['adj_atoms']
                charges = molecule['charges']
                # Charges can be either None, empty dict, or dict.
                if charges is None:
                    charges = {}
                atoms, A = _parse_adj_atoms(adj_atoms, charges)
                mapping = molecule['mapping']
                if 'atom_order' in molecule:
                    atom_order = np.array(molecule['atom_order'])
                    molecules[molecule_name] = Molecule(smiles, atoms, A, mapping, charges, atom_order)
                else:
                    molecules[molecule_name] = Molecule(smiles, atoms, A, mapping, charges)

    return molecules


class Simulation:
    """Class that represents the information of one simulation.

    Attributes
    ----------
    snapshot : str
        Absolute path to the pdb file.
    traj : str
        Absolute path to the xtc file.
    molecules : dict
        Information of all molecules of a simulation.
        Dictionary values can be either a dictionary itself or of type Molecule.
    time_step : int
        Snapshots of trajectories are analyzed every 'time_step' ps.
    """

    def __init__(self, t, s, rng, time_step=500, x=None):
        """
        Parameters
        ----------
        t : dict
            Information of all molecules of a simulation.
            Dictionary values can be either a dictionary itself or of type Molecule.
        s : str
            Absolute path to the pdb file.
        rng : np.random.default_rng()
            Default random number generator of numpy.
        time_step : int, default 500
            Snapshots of trajectories are analyzed every 'time_step' ps.
        x : str, default None
            Absolute path to the simulation trajectory file.
        """
        self.snapshot = s
        self.traj = x
        self.molecules = _parse_topology(t, rng)
        self.time_step = time_step
        self.sampling_step = None
        self.universe = None
        self.bonds_data_ = None
        self.angles_data = None

    def derive_topology(self):
        """
        Derive topology for all molecules.

        Loop over all molecules and derive their topologies:
            - Determine adjacency matrix of each fragment
            - Determine dihedrals of z-matrix based on the adjacency matrix
            - Derive the connectivity of the fragments and the connecting atoms
            - Derive loop order.
        Ignore water molecules.
        """
        for molecule in self.molecules.values():
            if not isinstance(molecule, Water):
                molecule.derive_topology()

    def read_simulation(self, *, memory=False):
        """
        Read the simulation data from disk and store it as attributes.
        Parameters
        ----------
        memory : bool
            True if the simulation data should be read in memory, False otherwise.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if self.traj is not None:
                self.universe = mda.Universe(self.snapshot, self.traj, in_memory=memory)
                delta_t = _extract_timestep(self.universe.trajectory)
                if delta_t > self.time_step:
                    logger = setup_logger(__name__)
                    logger.warning(f'Your requested time step is {self.time_step}, which is smaller than the '
                                   f'time step of the provided simulation, namely {delta_t}. Data will be extracted'
                                   f' with a time step of {delta_t}.')
                    self.sampling_step = int(delta_t)
                else:
                    self.sampling_step = int(self.time_step / delta_t)
            else:
                self.universe = mda.Universe(self.snapshot, in_memory=memory)

    def extract_fr_data(self):
        """
        Extract data from the simulation trajectory.

        For each fragment pair the internal coordinates of both fragments and the connector is extracted.
        Additionally, features (currently center of mass distance between both fragments) are extracted.
        If the residue is a water molecule, it is skipped.
        """
        if self.sampling_step is None:
            logger = setup_logger(__name__)
            logger.error('Sampling step is not specified. Maybe you did not provide a trajectory?')
            sys.exit(-1)
        for _ in self.universe.trajectory[::self.sampling_step]:
            for residue in self.universe.residues:
                if isinstance(self.molecules[residue.resname], Water):
                    continue
                else:
                    molecule = self.molecules[residue.resname]
                    if len(molecule.fragments) > 1:
                        for fr1_name, fr2_name in molecule.loop_order:
                            fr_pair = molecule.fr_pairs[(fr1_name, fr2_name)]
                            fr_pair.extract_fr_data(residue)
                    else:
                        for _, fr in molecule.fragments.items():
                            fr.extract_fr_data(residue)

    def extract_bond_data(self):
        """
        Extract bond lengths from the simulation trajectory.

        First, for each molecule all bonds are identified, i.e. atom names and bond types.
        Second, the bond lengths are calculated and stored as attributes.
        """
        # Determine all bonds for each molecule
        bonds_idx = {}
        for mol_name, mol in self.molecules.items():
            if isinstance(mol, Water):
                continue
            else:
                bonds_idx[mol_name] = _bonds_idx_mol(mol.atoms_f, mol.elements_f, mol.bond_list_f)

        # Extract bond values
        self.bonds_data_ = {}
        if self.sampling_step is None:
            logger = setup_logger(__name__)
            logger.error('Sampling step is not specified. Maybe you did not provide a trajectory?')
            sys.exit(-1)
        for _ in self.universe.trajectory[::self.sampling_step]:
            for residue in self.universe.residues:
                if isinstance(self.molecules[residue.resname], Water):
                    continue
                else:
                    for idx, atoms in bonds_idx[residue.resname].items():
                        for atom_list in atoms:
                            bond_d = residue.atoms.select_atoms(mda_selection(atom_list)).bond.value()
                            if idx in self.bonds_data_:
                                if len(self.bonds_data_[idx]) < 1000:
                                    self.bonds_data_[idx].append(bond_d)
                            else:
                                self.bonds_data_[idx] = [bond_d]

    def extract_angle_data(self):
        """
        Extract angles from the simulation trajectory.

        First, for each molecule all angles are identified, i.e. atom names and bond types.
        Second, the angles are calculated and stored as attributes.
        """
        # Determine all angles for each molecule
        angles_idx = {}
        for mol_name, mol in self.molecules.items():
            if isinstance(mol, Water):
                continue
            else:
                angle_list = derive_angle_list(mol.A_f.values)
                angles_idx[mol_name] = _angles_idx_mol(mol.atoms_f, mol.elements_f, angle_list)

        # Extract angle values
        self.angles_data_ = {}
        if self.sampling_step is None:
            logger = setup_logger(__name__)
            logger.error('Sampling step is not specified. Maybe you did not provide a trajectory?')
            sys.exit(-1)
        for _ in self.universe.trajectory[::self.sampling_step]:
            for residue in self.universe.residues:
                if isinstance(self.molecules[residue.resname], Water):
                    continue
                else:
                    for idx, atoms in angles_idx[residue.resname].items():
                        for atom_list in atoms:
                            mda_atoms = residue.atoms.select_atoms(mda_selection(atom_list))
                            mda_atoms_reordered = reorder_atom_group(mda_atoms, np.array(atom_list))
                            angle = calc_angle(*mda_atoms_reordered.positions)
                            if idx in self.angles_data_:
                                if len(self.angles_data_[idx]) < 1000:
                                    self.angles_data_[idx].append(angle)
                            else:
                                self.angles_data_[idx] = [angle]

    def write_to_db(self, database, ignore_fr_pairs=None, ignore_fr=None):
        """
        Write bond, angle, and molecule data extracted from a simulation trajectory to the database.

        Parameters
        ----------
        database : DBdata
        ignore_fr_pairs: list, default None
            Contains the identifiers of fragment pairs that should be ignored.
        ignore_fr: list, default None
            Contains the identifiers of fragments that should be ignored.
        """
        # Write bond and angle data from simulations to database
        database.add_bond_data(self.bonds_data_)
        database.add_angle_data(self.angles_data_)

        # Write molecule data from simulations to database
        for molecule in self.molecules.values():
            if not isinstance(molecule, Water):
                molecule.write_to_db(database, ignore_fr_pairs, ignore_fr)

    def load_models_db(self, database):
        """
        Load models for fragment pairs and fragments from the given database and store them as attributes.
        Parameters
        ----------
        database : DBdata
        """
        for molecule in self.molecules.values():
            if not isinstance(molecule, Water):
                if len(molecule.fragments) > 1:
                    for fr_pair_id, fr_pair in molecule.fr_pairs.items():
                        identifier, reverse = database.isin_fr_pair(fr_pair)
                        if identifier:
                            models_db = database.get_fr_pair_models(identifier, reverse)
                            if models_db:
                                fr_pair.set_models(*models_db)
                                fr_pair.reverse = reverse
                            else:
                                logger = setup_logger(__name__)
                                logger.error(f"No models in database for fragment pair {fr_pair_id}.")
                                sys.exit(-1)
                        else:
                            logger = setup_logger(__name__)
                            logger.error(f"Fragment pair {fr_pair_id} is not in the database.")
                            sys.exit(-1)
                else:
                    for fr_id, fr in molecule.fragments.items():
                        identifier = database.isin_fr(fr)
                        if identifier:
                            models_db = database.get_fr_models(identifier)
                            if models_db:
                                fr.set_models(*models_db)
                            else:
                                logger = setup_logger(__name__)
                                logger.error(f"No models in database for fragment {fr_id}.")
                                sys.exit(-1)
                        else:
                            logger = setup_logger(__name__)
                            logger.error(f"Fragment {fr_id} is not in the database.")
                            sys.exit(-1)

    def backmap(self, database, rng, hydrogens=False):
        """
        Backmap a coarse-grained snapshot to atomistic resolution.

        For each molecule of the coarse-grained structure the following steps are performed:
            1. Predict the conformation of each fragment based on the conformation of the coarse grained molecule.
            2. Translate the conformations to the positions of the respective coarse-grained beads.
            3. Optimize the connectors to properly connect fragments:
                - Fragments are rotated and translated such that accurate bond lengths, angles,
                  and dihedral angles are obtained for the connector.
                - Water and one bead molecules are rotated such that no clashes to neighboring atoms occur.
            4. Hydrogens are added to molecules with more than one bead.
            5. Clashes are resolved by shifting individual atoms. -> Necessary for subsequent energy minimization.

        Parameters
        ----------
        database : DBdata
            Database storing data on fragment pairs, fragments, bonds, and angles.
        rng : np.random.default_rng()
            Default random number generator of numpy.
        hydrogens : bool
            True if hydrogens should be included in the backmapped atomistic structure, False otherwise.

        Returns
        -------
        MDAnalysis.AtomGroup
            Backmapped atomistic structure.
        """
        cg = self.universe
        aa = _atomistic_universe(cg, self).atoms
        aa_coords = np.full((aa.n_atoms, 3), np.inf)
        aa.positions = aa_coords
        counter = 0
        water = None
        logger = setup_logger(__name__)
        logger.info('Prediction and optimization')
        for i, cg_residue in enumerate(cg.residues):
            if i % 1000 == 0 and i != 0:
                logger.info(f'Finished {i} molecules.')
            molecule = self.molecules[cg_residue.resname]
            if isinstance(molecule, Water):
                water = cg_residue.resname
                conf, d_max = molecule.predict_confs(rng)
                bead_coord = cg_residue.atoms.positions
                conf += (bead_coord - center_of_mass(conf, molecule.masses))
                coords_neighbors = aa.select_atoms(f'point {bead_coord[0][0]} {bead_coord[0][1]} {bead_coord[0][2]} '
                                                   f'{d_max + 1.5}').positions
                conf = optimize_one_bead_mol(conf, coords_neighbors, bead_coord, aa.dimensions,
                                             options={'eps': 1e-5}, rng=rng)
                aa_coords[counter: counter + molecule.n_atoms] = conf.copy()
            elif len(molecule.fragments) == 1:
                fr = next(iter(molecule.fragments.values()))
                conf = fr.predict_confs(rng)
                bead_coord = cg_residue.atoms.positions
                conf += (bead_coord - center_of_mass(conf, fr.masses))
                conf = rotate_random(conf, cg_residue.atoms.positions, rng)
                d_max = sphere_radius(conf, bead_coord)
                coords_neighbors = aa.select_atoms(f'point {bead_coord[0][0]} {bead_coord[0][1]} {bead_coord[0][2]} '
                                                   f'{d_max + 1.5}').positions
                conf = optimize_one_bead_mol(conf, coords_neighbors, bead_coord, aa.dimensions,
                                             options={'eps': 1e-5}, rng=rng)
                conf = conf[molecule.atom_order_idx]
                aa_coords[counter: counter + molecule.n_atoms] = conf.copy()
            else:
                for fr1_name, fr2_name in molecule.loop_order:
                    fr_pair = molecule.fr_pairs[(fr1_name, fr2_name)]
                    fr1 = fr_pair.fr1
                    fr2 = fr_pair.fr2
                    if fr_pair.fr1.pred_internal is None:
                        conf1, internal1, conf2, internal2, dihedral = fr_pair.predict_confs(cg_residue, rng)
                        fr1.pred_internal = internal1

                        # Set predicted fragment to CG coords
                        cg_coord1 = cg_residue.atoms.select_atoms(f'name {fr1_name}').positions
                        if cg_coord1.size == 0:
                            logger.error(f'I can not extract the CG coordinates for {fr1_name}.'
                                         f'Do the bead names in your mapping file and your CG structure match?')
                            sys.exit(-1)
                        conf1 += (cg_coord1 - center_of_mass(conf1, fr1.masses_f))
                        fr1.pred_coords = conf1.copy()
                    else:
                        _, _, conf2, internal2, dihedral = fr_pair.predict_confs(cg_residue, rng,
                                                                                    pred_internal1=fr1.pred_internal)

                    fr2.pred_internal = internal2
                    fr_pair.pred_dihedral = dihedral[0]

                    # Set predicted fragment to CG coords
                    cg_coord2 = cg_residue.atoms.select_atoms(f'name {fr2_name}').positions
                    if cg_coord2.size == 0:
                        logger.error(f'I can not extract the CG coordinates for {fr2_name}.'
                                     f'Do the bead names in your mapping file and your CG structure match?')
                        sys.exit(-1)
                    conf2 += (cg_coord2 - center_of_mass(conf2, fr2.masses_f))
                    fr2.pred_coords = conf2.copy()

                optimize_molecule(molecule, database)

                aa_coords_residue = np.full((molecule.n_atoms_f, 3), np.inf)
                for fr in molecule.fragments.values():
                    aa_coords_residue[fr.atoms_idx_f] = fr.pred_coords

                if hydrogens and len(molecule.fragments) > 1:
                    aa_coords_residue = correct_hydrogens(molecule, aa_coords_residue)
                else:
                    h_coords = np.full((molecule.n_atoms - molecule.n_atoms_f, 3), np.inf)
                    aa_coords_residue = np.vstack((aa_coords_residue, h_coords))

                aa_coords_residue = aa_coords_residue[molecule.atom_order_idx]
                aa_coords[counter: counter + molecule.n_atoms] = aa_coords_residue.copy()

                for fr in molecule.fragments.values():
                    fr.reset_labels()
            counter += molecule.n_atoms
            aa.positions = aa_coords  # Update coords for water selection

        # Correct indices for water molecules
        if water is not None:
            aa = correct_water(aa, water).atoms

        # Remove hydrogens
        if not hydrogens:
            idx = idx_atoms_f(aa.names)
            aa_coords = aa_coords[idx]
            aa = aa[idx]

        logger.info('Resolve clashing atoms.')
        # Resolve atom clashes
        clashing_atoms(aa_coords, aa.dimensions, rng, radius=0.15)
        aa.positions = aa_coords

        return aa


def _atomistic_universe(cg, snapshot):
    """
    Initialize atomistic universe that later represents the backmapped structure.

    Parameters
    ----------
    cg : MDAnalysis.Universe
        The coarse grained structure that gets backmapped. Residue names and box dimensions are extracted.
    snapshot : Simulation
        Simulation object storing information on the mapping of coarse-grained to atomistic resolution.
    Returns
    -------
    MDAnalysis.Universe
        Initialized atomistic universe.
    """
    residue_names = cg.residues.resnames
    n_residues = residue_names.size
    atoms = np.array([snapshot.molecules[mol].atom_order for mol in residue_names], dtype=object)
    residue_idx = np.array([count for count, sublist in enumerate(atoms) for _ in sublist])
    atoms = np.concatenate(atoms)
    n_atoms = atoms.size
    u = mda.Universe.empty(n_atoms, n_residues=n_residues, atom_resindex=residue_idx, trajectory=True)
    u.add_TopologyAttr('name', atoms)
    u.add_TopologyAttr('resname', residue_names)
    u.add_TopologyAttr('resid', list(range(1, n_residues + 1)))
    u.dimensions = cg.dimensions
    return u


def _bonds_idx_mol(atoms, elements, bond_list):
    """
    Determines all unique bonds of a molecule.

    Desired output format is of a bond is (element1, element2, bond type), e.g. ('C', 'C', 1) indicating
    a single bond between two carbon atoms. For each bond a corresponding list of atoms names is derived,
    e.g. [['C1', 'C2'], ['C2', 'C3']], indicating a single bond between atoms with the names 'C1' and 'C2', and
    'C2' and 'C3'. Thus, bond types are linked to specific atom names in the trajectory.

    Parameters
    ----------
    atoms : numpy.ndarray
        Atom names.
    elements : numpy.ndarray
        Atom elements.
    bond_list : numpy.ndarray
        Bond list. Each element contains three values, namely two indices and a bond type, e.g. [1, 4, 2].
        This specifies that atoms with index 1 and 4 in 'atoms' are connected via a double bond (2).

    Returns
    -------
    dict
        Unique bonds. e.g. {('C', 'C', 1): [['C1', 'C2'], ['C2', 'C3']]}
    """
    # for each bond extract: element of atom1 and atom2, bond type, name in simulation of atom1 and atom2
    a1 = atoms[bond_list[:, 0]]
    a2 = atoms[bond_list[:, 1]]
    e1 = elements[bond_list[:, 0]]
    e2 = elements[bond_list[:, 1]]
    bond_type = bond_list[:, 2]
    bonds = np.column_stack((e1, e2, bond_type, a1, a2))

    # Alphabetically order the first two elements: C-O same as O-C
    idx = bonds[:, 1] < bonds[:, 0]
    if np.any(idx):
        bonds[idx, :2] = bonds[idx, 1::-1]

    # Group identical bonds
    bonds_unique = {}
    for bond in bonds:
        key = tuple(bond[:3])
        if key in bonds_unique:
            bonds_unique[key].append(list(bond[3:]))
        else:
            bonds_unique[key] = [list(bond[3:])]
    return bonds_unique


def _angles_idx_mol(atoms, elements, angle_list):
    """
    Determines all unique angles of a molecule.

    Desired output format is of an angle is (element1, element2, element3, bond type 1, bond type 2),
    e.g. ('C', 'C', 'O', 1, 2) indicating a single bond between two carbon atoms
    and a double bond between carbon and oxygen. For each angle a corresponding list of atoms names is derived,
    e.g. [['C1', 'C2', 'O'], ['C4', 'C5', 'O2']], indicating for the first element that a single bond between atoms
    with the names 'C1' and 'C2', and a double bond between 'C2' and 'O' exists.
    Thus, bond types are linked to specific atom names in the trajectory.

    Parameters
    ----------
    atoms : numpy.ndarray
        Atom names.
    elements : numpy.ndarray
        Atom elements.
    angle_list : numpy.ndarray
        Angle list. Each element contains five values, namely three indices and two bond types, e.g. [1, 4, 8, 1, 2].
        This specifies that atoms with indices 1 and 4 in 'atoms' are connected via a single bond (1) and atoms with
        indices 4 and 8 in 'atoms' are connected via a double bond (2).

    Returns
    -------
    dict
        Unique angles. e.g. {('C', 'C', 'O', 1, 2): [['C1', 'C2', 'O'], ['C4', 'C5', 'O2']]}
    """
    # for each bond extract: element of atom1 and atom2, bond type, name in simulation of atom1 and atom2
    a1 = atoms[angle_list[:, 0]]
    a2 = atoms[angle_list[:, 1]]
    a3 = atoms[angle_list[:, 2]]
    e1 = elements[angle_list[:, 0]]
    e2 = elements[angle_list[:, 1]]
    e3 = elements[angle_list[:, 2]]
    bond_types = angle_list[:, 3:]
    angles = np.column_stack((e1, e2, e3, bond_types, a1, a2, a3))

    # Alphabetically order angles: C-C-O same as O-C-C
    idx = angles[:, 2] < angles[:, 0]
    if np.any(idx):
        angles[idx, 0], angles[idx, 2] = angles[idx, 2], angles[idx, 0]
        angles[idx, 5], angles[idx, 7] = angles[idx, 7], angles[idx, 5]
        angles[idx, 3], angles[idx, 4] = angles[idx, 4], angles[idx, 3]

    # Group identical angles
    angles_unique = {}
    for angle in angles:
        key = tuple(angle[:5])
        value = list(angle[5:])
        if key in angles_unique and value not in angles_unique[key]:
            angles_unique[key].append(value)
        else:
            angles_unique[key] = [value]

    return angles_unique
