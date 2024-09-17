import sys

from MDAnalysis.topology.guessers import guess_masses, guess_types
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

from artsm.topology.fragment_pair import FragmentPair, Connector
from artsm.topology.fragment import Fragment
from artsm.utils.bond_angle_lists import derive_bond_list
from artsm.utils.containers import extract_atoms_f, extract_atoms_h, remove_duplicates, idx_atoms_f, element_idx
from artsm.utils.other import setup_logger, numpy_pairwise_combinations
from artsm.utils.smiles import generate_rdkit_mol


def _check_stereo(mol_stereo):
    """
    Check if a molecule has unspecified stereocenters.

    Prints a warning if stereocenters are present that are not specified in the provided SMILES.

    Parameters
    ----------
    mol_stereo : rdkit.Chem.rdchem.Mol
        RDKit molecule.
    """
    pot_stereo = Chem.FindPotentialStereo(mol_stereo)
    for element in pot_stereo:
        # element has attributes type, centeredOn, specified, descriptor, controllingAtoms
        if element.specified == Chem.StereoSpecified.Unspecified:
            logger = setup_logger(__name__)
            logger.warning(f'Potential unspecified stereocenter of type {element.type} for '
                           f'SMILES {Chem.MolToSmiles(mol_stereo)}.'
                           f'Please specify cis/trans as well as chirality to provide accurate results.')


def _parse_rdkit_mol(smiles, atoms):
    """
    Parses a SMILES string to a RDKit molecule.
    Parameters
    ----------
    smiles : str
        SMILES string.
    atoms : np.ndarray
        Atoms of the molecule.
    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    _check_stereo(mol)  # Check if molecule has unspecified stereocenters
    rank = list(Chem.CanonicalRankAtoms(mol, includeChirality=False))
    # reorder atoms according to canonical rank and store the mapping 
    # (H atoms are not included in the mapping)
    idx_to_name = {idx: name for idx, name in enumerate(atoms[rank])}
    # add monomer info to atoms in mol object using the mapping
    for atom in mol.GetAtoms():
        info = Chem.AtomMonomerInfo()
        info.SetName(idx_to_name[atom.GetIdx()])
        atom.SetMonomerInfo(info)
    return mol


def _parse_mapping(mapping, A):
    """
    Derive fragments of a molecule and the corresponding atoms and connectivity.

    Parameters
    ----------
    mapping : dict
        Contains information of the field 'mapping' in a simulation config file. Specifies the mapping from
        coarse-grained to atomistic.
    A : pd.DataFrame
        Adjacency matrix of molecule.
        Index: Atom names
        Columns: Atom names

    Returns
    -------
    dict
        Dictionary with keys as fragment names and values as Instances of class Fragment
    """
    atoms_mol = A.index.to_numpy()
    fragments = {}
    for fr_name, fr_atoms in mapping.items():
        # Instantiate the Fragment. Atom order will be changed in the __init__ method of Fragments.
        fr_A = A.loc[fr_atoms][fr_atoms].to_numpy()
        fr = Fragment(np.array(fr_atoms), fr_A, name=fr_name)
        # Get the positions of the fragments atoms in the molecule. Important for correct output later.
        fr.atoms_idx = element_idx(fr.atoms, atoms_mol)
        fr.atoms_idx_f = element_idx(fr.atoms_f, atoms_mol)
        fragments[fr_name] = fr
    return fragments


def _fragment_connected(fr1_atoms, fr2_atoms, A_molecule):
    """
    Return True if two fragments are connected, False otherwise.

    Parameters
    ----------
    fr1_atoms : numpy.ndarray
        Atoms of fragment 1.
    fr2_atoms : numpy.ndarray
        Atoms in fragment 2.
    A_molecule : pd.Dataframe
        Adjacency matrix of the molecule containing fragment 1 and fragment 2.
        Index: Atom names
        Columns: Atom names

    Returns
    -------
    bool
        True if fragments are connected, False otherwise.
    """
    A_fr_pair = A_molecule.loc[fr1_atoms][fr2_atoms]
    if A_fr_pair.sum().sum() > 0:
        return True
    else:
        return False


def _fragment_connections(fr1_atoms, fr2_atoms, A_molecule):
    """
    Return two atoms that connect two fragments (empty list if fragments are not connected).
    Parameters
    ----------
    fr1_atoms : numpy.ndarray
        Atom names of fragment 1.
    fr2_atoms : numpy.ndarray
        Atom names of fragment 2.
    A_molecule : pd.DataFrame
        Adjacency matrix of the molecule.
        Index: Atom names
        Columns: Atom names
    Returns
    -------
    numpy.ndarray
        Atoms that connect fragment 1 with fragment 2, e.g. [['C1', 'C2']]. The first atom belongs to fragment 1
        and the second atom to fragment 2. Empty array is returned if fragments are not connected.
    """
    A_connection = A_molecule.loc[fr1_atoms][fr2_atoms]
    idx = np.where(A_connection >= 1)
    row = A_connection.index[idx[0]]
    column = A_connection.columns[idx[1]]
    return np.column_stack((row, column))


def _extend_connection(fr1_atoms, fr2_atoms, A_f, connection):
    """
    Extend the connection of two fragments from two to four atoms.
    Parameters
    ----------
    fr1_atoms : numpy.ndarray
        Atom names of fragment 1.
    fr2_atoms : numpy.ndarray
        Atom names of fragment 2.
    A_molecule : pd.DataFrame
        Adjacency matrix of the molecule.
        Index: Atom names
        Columns: Atom names
    connection : numpy.ndarray
        The atoms connecting fragment 1 with fragment 2, e.g. ['C1', 'C2'].
    Returns
    -------
    numpy.ndarray
        Atoms that connect fragment 1 with fragment 2 and two atoms bonded to these connecting atoms,
        e.g. [['C1', 'C2', 'C3', 'C4']], with C2 and C3 connecting the fragments and
        C1 being bound to C2, C4 being bound to C3.
    """
    connection_extended = connection.copy()
    atom1, atom2 = connection
    # Extend atom of fr1
    candidates = A_f.columns[A_f.loc[atom1] >= 1].values
    candidates = candidates[np.isin(candidates, fr1_atoms)]
    candidates = candidates[np.invert(np.isin(candidates, connection))]
    connection_extended = np.insert(connection_extended, 0,
                                candidates[0])
    # Extend atom of fr2
    candidates = A_f.index[A_f[atom2] >= 1].values
    candidates = candidates[np.isin(candidates, fr2_atoms)]
    candidates = candidates[np.invert(np.isin(candidates, connection))]
    connection_extended = np.insert(connection_extended, 3,
                                candidates[0])
    return connection_extended


class Molecule:
    """Class that represents the information of one molecule."""

    def __init__(self, smiles, atoms, A, mapping, charges, atom_order=None):
        """
        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.
        atoms : numpy.ndarray
            Atoms of molecule, e.g. np.array(['C1', 'C2']).
        A : numpy.ndarray
            Adjacency matrix of the molecule.
            Index: Atom names
            Columns: Atom names
        mapping : dict
            Mapping of atoms to CG beads, e.g. 'C1O': ['C1', 'C2'], 'C2A': ['C3', 'C4'].
        atom_order : numpy.ndarray, default None
            Order of atoms in the molecule, e.g. np.arr(['C1', 'C2', 'C3', ...]). Required to force a specific order
            of the atoms when writing to file.
        """
        self.atoms = atoms
        self.atoms_f = extract_atoms_f(atoms)
        self.atoms_h = extract_atoms_h(atoms)

        self.n_atoms = atoms.size
        self.n_atoms_f = self.atoms_f.size
        self.n_atoms_h = self.atoms_h.size

        self.elements = guess_types(atoms)
        self.elements_f = guess_types(self.atoms_f)
        self.elements_h = guess_types(self.atoms_h)

        self.masses = guess_masses(self.elements)
        self.masses_f = guess_masses(self.elements_f)
        self.masses_h = guess_masses(self.elements_h)

        self.charges = charges

        if atom_order is not None:
            self.atom_order = atom_order 
            # if atom_order is [C1, C2, C3, C4] and atoms is [C2, C1, C4, C3]
            # then atom_order_idx is [1, 0, 3, 2]
            self.atom_order_idx = element_idx(atom_order, atoms)
        self.smiles = Chem.CanonSmiles(smiles)

        self.A = pd.DataFrame(A, index=atoms, columns=atoms)
        self.A_f = self.A.loc[self.atoms_f][self.atoms_f]
        self.A_h = self.A.loc[self.atoms_h][self.atoms_h]

        self.bond_list = derive_bond_list(A)
        self.bond_list_f = derive_bond_list(self.A_f.values)

        self.fragments = _parse_mapping(mapping, self.A)
        self.rdkit_mol = _parse_rdkit_mol(self.smiles, self.atoms)  # Stores stereochemistry info
        self.fr_pairs = None
        self.fragments_A = None
        self.loop_order = None
        self.loop_order_flat = None

    def derive_topology(self):
        """
        Derive the topology of the molecule with respect to the CG model. This includes:
            - Determine adjacency matrix of each fragment
            - Determine dihedrals of z-matrix based on the adjacency matrix
            - Derive the connectivity of the fragments and the connecting atoms
            - Derive loop order.
        """
        self._derive_dihedrals()
        if len(self.fragments) > 1:
            self._derive_fragments_A()
            self._derive_loop_order()
            self._derive_fragment_pairs()
            self._derive_smiles()
        else:
            for _, fr in self.fragments.items():
                fr.smiles = self.smiles

    def _derive_fragments_A(self):
        """
        Derive the fragments_A DataFrame based on the fragments and connectivity matrix A.

        This method calculates the connectivity between fragments and populates the fragments_A DataFrame
        with 1s for connected fragments and 0s for non-connected fragments. Called by ~self.derive_topology.
        """
        fr_names = list(self.fragments.keys())
        self.fragments_A = pd.DataFrame(0, index=fr_names, columns=fr_names)
        fragment_pairs = numpy_pairwise_combinations(np.array(list(self.fragments.values())))

        for (fr1, fr2) in fragment_pairs:
            if _fragment_connected(fr1.atoms, fr2.atoms, self.A):
                self.fragments_A[fr1.name][fr2.name] = 1
                self.fragments_A[fr2.name][fr1.name] = 1

    def _derive_fragment_pairs(self):
        """
        Determine the connectivity of fragments and subsequently fragment pairs.

        Called by ~self.derive_topology
        """
        self.fr_pairs = {}
        for fr1_name, fr2_name in self.loop_order:
            if (fr1_name, fr2_name) not in self.fr_pairs:
                fr1 = self.fragments[fr1_name]
                fr2 = self.fragments[fr2_name]
                connections = _fragment_connections(fr1.atoms_f, fr2.atoms_f, self.A_f)
                if len(connections) == 0:
                    continue
                elif len(connections) > 1:
                    logger = setup_logger(__name__)
                    logger.warning(f'I found {len(connections)} bonds between {fr1.name} '
                                   f'and {fr2.name} in molecule with SMILES {self.smiles}. '
                                   f'I will only use the bond {connections[0]} to build the fragment pair.')

                # Derive connecting atoms
                connection = connections[0]
                connector_atoms = _extend_connection(fr1.atoms_f, fr2.atoms_f, self.A_f, connection)
                A_connector = self.A_f[connector_atoms].loc[connector_atoms].values
                bond_list = np.array([A_connector[0, 1], A_connector[1, 2], A_connector[2, 3]])
                con = Connector(connector_atoms, bond_list)

                self.fr_pairs[(fr1.name, fr2.name)] = FragmentPair(fr1, fr2, con)

    def _derive_dihedrals(self):
        """
        Derives the dihedral angles for each fragment in the molecule.

        This method iterates over the fragments in the molecule and calls the ``self.derive_internal_coords` method
        for each fragment to calculate the dihedral angles. Called by ~self.derive_topology.
        """
        for fr in self.fragments.values():
            fr.derive_internal_coords()

    def _derive_loop_order(self):
        """
        Derive the loop order for fragments in the molecule.

        First, a terminal fragment is determined, added to the loop_order_list, and expanded, i.e. neighbor fragments
        are identified. Afterwards, the neighbor fragments are expanded in random order. This is repeated until all
        fragments are included in the loop order. Called from ~self.derive_topology.
        """
        self.loop_order = np.empty((len(self.fragments) - 1, 2), dtype=object)
        # Select a terminal fragment as starting point.
        if self.fragments_A.sum().sum() == 0:
            logger = setup_logger(__name__)
            logger.error('Your fragments seem to be not connected, hinting at a molecule that is not connected.'
                         'Please recheck your mapping file.')
            sys.exit(-1)
        elif (self.fragments_A.sum(axis=1) == 1).sum() == 0:
            # Cyclic molecule, no terminal fragment
            fr_start = self.fragments_A.loc[self.fragments_A.sum(axis=1) == 2].index[0]
        else:
            # choose the first terminal fragment as the start
            fr_start = self.fragments_A.loc[self.fragments_A.sum(axis=1) == 1].index[0]
        expansion_order = [fr_start]
        queue = [fr_start]

        # Might fail for atoms that have a ring and a chain structure
        while len(queue) > 0:
            fr = queue[-1]
            fragments_connected = self.fragments_A.loc[self.fragments_A[fr] == 1].index.values
            candidates = fragments_connected[np.invert(np.isin(fragments_connected, expansion_order))]
            if candidates.size <= 1:
                queue = queue[:-1]
            if candidates.size > 0:
                fr_next = candidates[0]
                queue.append(fr_next)
                expansion_order.append(fr_next)
                self.loop_order[len(expansion_order) - 2] = np.array([fr, fr_next])

        order = self.loop_order.flatten()
        self.loop_order_flat = remove_duplicates(order)

    def _derive_smiles(self):
        """
        Derives the SMILES string for each fragment and fragment pair in the molecule.

        Iterates over the fragment pairs in the molecule and calls the `derive_smiles` method
        for each fragment and fragment pair to calculate the SMILES string. Called by ~.self.derive_topology.
        """
        atom_to_idx = {atom.GetMonomerInfo().GetName(): atom.GetIdx() for atom in self.rdkit_mol.GetAtoms()}
        if len(self.fr_pairs) == 1:
            for _, con in self.fr_pairs.items():
                con.smiles = self.smiles
                # Get smiles for fragments of fr_pairs
                idx1 = atom_to_idx[con.con.atoms[1]]
                idx2 = atom_to_idx[con.con.atoms[2]]
                bond = self.rdkit_mol.GetBondBetweenAtoms(idx1, idx2)
                mol_fragmented = Chem.FragmentOnBonds(self.rdkit_mol, (bond.GetIdx(),), addDummies=False)
                fragments2 = Chem.GetMolFrags(mol_fragmented, asMols=True)
                for fr2 in fragments2:
                    atom_name = fr2.GetAtomWithIdx(0).GetMonomerInfo().GetName()
                    if atom_name in con.fr1.atoms:
                        con.fr1.smiles = Chem.MolToSmiles(fr2)
                    elif atom_name in con.fr2.atoms:
                        con.fr2.smiles = Chem.MolToSmiles(fr2)
        # Get mapping of atom names to atom indices
        else:
            for fr_pair in self.fr_pairs.values():
                atoms_fr_pair = np.concatenate((fr_pair.fr1.atoms, fr_pair.fr2.atoms))
                bonds = np.column_stack((self.atoms[self.bond_list[:, 0]], self.atoms[self.bond_list[:, 1]]))
                bonds_con_mol = []
                for bond in bonds:
                    if ((bond[0] in atoms_fr_pair and bond[1] not in atoms_fr_pair)
                            or (bond[1] in atoms_fr_pair and bond[0] not in atoms_fr_pair)):
                        bonds_con_mol.append(bond)
                bonds_remove = []
                for a1, a2 in bonds_con_mol:
                    idx1 = atom_to_idx[a1]
                    idx2 = atom_to_idx[a2]
                    bond = self.rdkit_mol.GetBondBetweenAtoms(idx1, idx2)
                    bonds_remove.append(bond.GetIdx())
                mol_fragmented = Chem.FragmentOnBonds(self.rdkit_mol, tuple(bonds_remove), addDummies=False)
                fragments = Chem.GetMolFrags(mol_fragmented, asMols=True)
                for fr in fragments:
                    if fr.GetAtomWithIdx(0).GetMonomerInfo().GetName() in atoms_fr_pair:
                        fr_pair.smiles = Chem.MolToSmiles(fr)
                        # Get smiles for fragments of fr_pairs
                        idx1 = atom_to_idx[fr_pair.con.atoms[1]]
                        idx2 = atom_to_idx[fr_pair.con.atoms[2]]
                        bond = mol_fragmented.GetBondBetweenAtoms(idx1, idx2)
                        mol_fragmented = Chem.FragmentOnBonds(mol_fragmented, (bond.GetIdx(), ), addDummies=False)
                        fragments2 = Chem.GetMolFrags(mol_fragmented, asMols=True)
                        for fr2 in fragments2:
                            atom_name = fr2.GetAtomWithIdx(0).GetMonomerInfo().GetName()
                            if atom_name in fr_pair.fr1.atoms:
                                fr_pair.fr1.smiles = Chem.MolToSmiles(fr2)
                            elif atom_name in fr_pair.fr2.atoms:
                                fr_pair.fr2.smiles = Chem.MolToSmiles(fr2)
                        break

    def write_to_db(self, db, ignore_fr_pairs=None, ignore_fr=None):
        """
        Write the extracted simulation data to a database.

        For fragments and fragment pairs, it is checked if they already exist in the database. If so,
        their data is appended to the existing data. Otherwise, general information on the fragments or fragment pairs
        are added to the database first.

        Parameters
        ----------
        db: DBdata
            The database object.
        ignore_fr_pairs: list, default None
            Contains the identifiers of fragment pairs that should be ignored.
        ignore_fr: list, default None
            Contains the identifiers of fragments that should be ignored.
        """
        if len(self.fragments) > 1:
            # Write fr_pair data from simulations to database
            for fr_pair in self.fr_pairs.values():
                identifier, reverse = db.isin_fr_pair(fr_pair)
                if identifier:
                    if ignore_fr_pairs is not None and identifier in ignore_fr_pairs:
                        continue
                    else:  # in db, not in ignore_fr_pairs so data added in prev iteration
                        db.append_fr_pair_data(identifier, fr_pair.get_data(), reverse)
                else:
                    db.add_fr_pair(fr_pair)
        else:
            # Write single bead data from simulations to database
            for fr in self.fragments.values():
                identifier = db.isin_fr(fr)
                if identifier:
                    if ignore_fr is not None and identifier in ignore_fr:
                        continue
                    else:
                        db.append_fr_data(identifier, fr.get_data())
                else:
                    db.add_fragment(fr)

