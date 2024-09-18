import os
import sys

from MDAnalysis.topology.guessers import guess_masses, guess_types
import numpy as np
import pandas as pd

from artsm.model.mainconfs import main_conformations
from artsm.model.training import probabilities_one_bead
from artsm.utils.angles import calc_angle, calc_dihedral
from artsm.utils.bond_angle_lists import derive_bond_list
from artsm.utils.containers import extract_atoms_f, extract_atoms_h
from artsm.utils.fileparsing import write_smiles
from artsm.utils.other import center_of_mass, mda_selection, setup_logger
from artsm.utils.smiles import canonical_atom_order


def _determine_candidates(atom, A, expansion_order, indices):
    """
    Determine atoms called candidates that are bonded to atom, occur in expansion order, and don't occur in indices.

    Called by _internal_coords_one_run.

    Parameters
    ----------
    atom : str
        The current atom for which the bonded atom candidates are determined.
    A : pandas.DataFrame
        Adjacency matrix of the molecule or fragment.
    expansion_order : list of str
        The order in which atoms were expanded.
    indices : list of str
        The indices of atoms in A that are excluded from being candidates.

    Returns
    -------
    numpy.ndarray
        Candidate atoms that are bonded to atom.
    """
    connected_atoms = A.loc[A[atom] >= 1].index.values
    # Only consider already expanded atoms.
    # filter the neighbors to only include atoms that are already in the expansion order but not in the indices.
    candidates = np.array([atom for atom in connected_atoms if atom in expansion_order
                           if atom not in indices])
    if candidates.size == 0:
        logger = setup_logger(__name__)
        logger.error('Determining dihedral indices failed.'
                     'Current atom has no connection to any previously processed atom.')
        sys.exit(-1)
    return candidates


def _internal_coords_one_run(zmatrix, A, expansion_order, queue):
    """
    Determines the dihedral angles of the zmatrix.

    Starting at a terminal atom, the molecule is traversed (only via bonded atoms) such that all atoms are visited.
    expansion_order stores the order in which atoms are visited.
    queue stores the atoms that need to be expanded.

    Initially, the starting atom is stored in queue and expansion_order.
    The following steps are performed until the queue is empty:
    1. Chose the last atom of the queue a1.
    2. Determine connected atoms that have not yet been visited (not in expansion_order).
       If one or less connected atoms are found, a1 is removed from the queue, because it is fully expanded.
    3. Choose one of the connected atoms a2. Add it to queue and expansion order.
    4. Repeat 1, 2, and 3 until the queue is empty

    In step 3: If the expansion order contains 4 or more atoms, the dihedral angle containing the latest atom added to
    the expansion order is determined and added to zmatrix.


    Parameters
    ----------
    zmatrix : list
        Dihedral angles of the zmatrix will be added to this list.
    A : pandas.DataFrame
        Adjacency matrix of the molecule or fragment.
    expansion_order : list of str
        List containing the starting atom. Will be modified.
    queue : list of str
        List containing the starting atom. Will be modified.
    """
    while len(queue) > 0:
        atom = queue[-1]
        # check which fragment atoms are connected to the queue atom
        connected_atoms = A.loc[A[atom] >= 1].index.values
        candidates = np.array([atom for atom in connected_atoms if atom not in expansion_order])
        if candidates.size <= 1:
            queue.pop()
        if candidates.size > 0:
            next_atom = candidates[0]
            queue.append(next_atom)
            expansion_order.append(next_atom)
            # Determine dihedral indices if more than 3 atoms have been queued.
            if len(expansion_order) > 3:
                indices = [next_atom]
                # Determine the first connected atom
                candidates = _determine_candidates(next_atom, A, expansion_order, indices)
                next_atom = candidates[-1]
                indices.append(next_atom)
                # Determine the second connected atom.
                candidates = _determine_candidates(next_atom, A, expansion_order, indices)
                if candidates.size == 1:
                    next_atom = candidates[0]
                    indices.append(next_atom)
                    # Determine the third connected atom.
                    candidates = _determine_candidates(next_atom, A, expansion_order, indices)
                    next_atom = candidates[-1]
                    indices.append(next_atom)
                else:
                    indices.extend([candidates[-1], candidates[-2]])
                indices = np.array(indices)
                if len(zmatrix) == 0 or not any(np.array_equal(dihedral, indices) for dihedral in zmatrix):
                    zmatrix.append(np.array(indices))


class Fragment:
    """Class that represents the information of one fragment."""

    def __init__(self, atoms, A, name=None, internal_coords=None, smiles=None):
        """
        Initializes the fragment object by setting the atoms, connectivity, and SMILES representation for
        stereochemistry.
        Atom elements and masses are guessed from the atom names.
        Bond lists are derived from the adjacency matrix.
        Attributes that are able to store information from simulations are initialized.
        
        Parameters
        ----------
        name : str
            Name of the fragment.
        atoms : np.ndarray
            Atom names of the fragment, e.g. np.arr(['C1', 'C2'])
        A : np.ndarray
            Adjacency matrix of the fragment. Value indicates the bond type.
        """
        if name is not None:
            self.name = name

        self.atoms = canonical_atom_order(atoms, A)
        self.atoms_f = extract_atoms_f(self.atoms)
        self.atoms_h = extract_atoms_h(self.atoms)
        self.atoms_idx = None  # indices of fragments atoms in the molecule
        self.atoms_idx_f = None

        self.n_atoms = self.atoms.size
        self.n_atoms_f = self.atoms_f.size
        self.n_atoms_h = self.atoms_h.size

        self.elements = guess_types(self.atoms)
        self.elements_f = guess_types(self.atoms_f)
        self.elements_h = guess_types(self.atoms_h)

        self.masses = guess_masses(self.elements)
        self.masses_f = guess_masses(self.elements_f)
        self.masses_h = guess_masses(self.elements_h)

        A_pd = pd.DataFrame(A, index=atoms, columns=atoms)
        self.A = A_pd.loc[self.atoms][self.atoms]
        self.A_f = self.A.loc[self.atoms_f][self.atoms_f]
        self.A_h = self.A.loc[self.atoms_h][self.atoms_h]

        self.zmatrix = internal_coords

        self.bond_list = derive_bond_list(A)
        self.bond_list_f = derive_bond_list(self.A_f.values)

        if smiles is not None:
            self.smiles = smiles

        # One bead molecules
        self._zmatrix_data = []
        self._coords_data = []
        self._main_coords = None
        self._model = None

        # Predicted conformations
        self.pred_internal = None
        self.pred_coords = None

    def derive_internal_coords(self):
        """
        Determine the internal coordinates of the fragment pair.

        Internal coordinates are:
            - Dihedral angles of the zmatrix if number of atoms >= 4
            - Angle if number of atoms == 3
            - Bond if number of atoms == 2
        Stored in self.zmatrix.
        """
        if self.n_atoms_f == 1:
            logger = setup_logger(__name__)
            logger.error('Sorry. Fragments containing only one atom are not supported yet.')
            sys.exit(-1)
        elif self.n_atoms_f == 2:
            self.zmatrix = [self.atoms_f]
        elif self.n_atoms_f == 3:
            # the sum method is used to count the number of non-zero elements in each row of the matrix.
            # The resulting Series contains the number of neighbors for each atom in the fragment.
            # finally, the selected indices correspond to the terminal atoms in the fragment.
            terminal_atoms = list(self.A_f.index[(self.A_f >= 1).sum(axis=1) == 1])
            middle_atom = [atom for atom in self.atoms_f if atom not in terminal_atoms][0]
            angle_atoms = np.array([terminal_atoms[0], middle_atom, terminal_atoms[1]])
            self.zmatrix = [angle_atoms]
        else:
            self.zmatrix = []
            # Select a terminal atom as starting point.
            starting_atoms = list(self.A_f.index[(self.A_f >= 1).sum(axis=1) == 1])

            while len(self.zmatrix) < self.n_atoms_f - 3:
                if len(starting_atoms) == 0:
                    logger = setup_logger(__name__)
                    logger.error(f'Sorry. I can not derive internal coordinates for fragment {self.name}.')
                    sys.exit()
                else:
                    starting_atom = starting_atoms.pop()
                    expansion_order = [starting_atom]
                    queue = [starting_atom]
                    _internal_coords_one_run(self.zmatrix, self.A_f, expansion_order, queue)

    def calc_zmatrix(self, residue, hydrogens=False):
        """
        Calculate the values for the internal coordinates given simulation data in form of a residue.
        Parameters
        ----------
        residue : MDAnalysis.core.groups.Residue
            Residue containing simulation data.
        hydrogens : bool, default False
            Whether the coordinates of hydrogens should be considered and stored in the database.

        Returns
        -------
        fr_zmatrix_data
            Values for zmatrix.
        fr_coords
            Extracted coordinates of the atoms in the simulation residue.
        """
        fr = residue.atoms.select_atoms(mda_selection(self.atoms))
        fr_coords = pd.DataFrame(fr.positions, index=fr.names)
        # Resolve the difference of atom ordering between MDAnalysis and ART-SM.
        # Exclude or include H atoms
        if hydrogens:
            fr_coords = fr_coords.loc[self.atoms]
        else:
            fr_coords = fr_coords.loc[self.atoms_f]
        fr_zmatrix_data = np.zeros(len(self.zmatrix))
        for i, atoms in enumerate(self.zmatrix):
            coords_atoms = fr_coords.loc[atoms].values
            if atoms.size == 2:
                fr_zmatrix_data[i] = np.linalg.norm(coords_atoms[0] - coords_atoms[1])
            elif atoms.size == 3:
                fr_zmatrix_data[i] = calc_angle(coords_atoms[0], coords_atoms[1], coords_atoms[2])
            else:
                fr_zmatrix_data[i] = calc_dihedral(coords_atoms[0], coords_atoms[1], coords_atoms[2], coords_atoms[3])

        return fr_zmatrix_data, fr_coords

    def reset_labels(self):
        """
        Resets the attributes pred_internal and pred_coords to None.

        Relevant for backmapping. Attributes have to be reset before predicting the conformations of the next molecule.
        """
        self.pred_internal = None
        self.pred_coords = None

    def com(self):
        """
        Calculate the center of mass of the fragment.

        Returns
        -------
        numpy.ndarray
            The center of mass of the fragment.
        """

        if self.pred_coords is None:
            return
        else:
            if self.pred_coords.shape[0] == self.n_atoms_f:
                return center_of_mass(self.pred_coords, self.masses_f)
            elif self.pred_coords.shape[0] == self.n_atoms:
                return center_of_mass(self.pred_coords, self.masses)
            else:
                logger = setup_logger(__name__)
                logger.error('Center of mass can not be predicted, since the number of coordinates do not match '
                             'the number of heavy atoms or of all atoms.')
                sys.exit(-1)

    def extract_fr_data(self, residue):
        """
        Extracts values fot the zmatrix and coordinates for a given residue in a simulation.

        Parameters
        ----------
        residue : MDAnalysis.core.groups.Residue
            Residue containing simulation data.
        """
        zmatrix_data, coords_data = self.calc_zmatrix(residue, hydrogens=True)
        self._zmatrix_data.append(zmatrix_data)
        self._coords_data.append(coords_data)

    def set_data(self, zmatrix_data, coords_data):
        """
        Sets the zmatrix and coordinates data of the fragment.

        Parameters
        ----------
        zmatrix_data : list
            The zmatrix data.
        coords_data : list
            The coordinates data.
        """
        self._zmatrix_data = zmatrix_data
        self._coords_data = coords_data

    def get_data(self):
        """Returns the zmatrix and coordinates data of the fragment."""
        return np.array(self._zmatrix_data), np.array(self._coords_data)

    def set_models(self, main_coords, model):
        """
        Sets the main coordinates and model of the fragment.

        Parameters
        ----------
        main_coords : list
            Coordinates of the main conformations.
        model : Model
            ML model. Requires at least methods fit and predict.
        """
        self._main_coords = main_coords
        self._model = model

    def get_models(self):
        """Returns the main coordinates and model of the fragment."""
        return self._main_coords, self._model

    def derive_models(self, path):
        """
        Derives the main conformations of the fragments via hierarchical clustering and their probabilities.

        Probabilities are stored in a model (OneBeadModel) to sample from the main conformations in backmapping.

        Parameters
        ----------
        path : str
            Path to output directory. Figures that visualize the clustering process and a file
            with the final probabilities is generated.
        """
        fr_path = os.path.join(path, 'mainconfs')
        write_smiles(os.path.join(fr_path, 'smiles.txt'), self.smiles)
        labels, representative_idx = main_conformations(self._zmatrix_data, self.zmatrix, fr_path)
        self._main_coords = self._coords_data[representative_idx]

        training_path = os.path.join(path, 'ml')
        self._model = probabilities_one_bead(labels, training_path)

    def predict_confs(self, rng):
        """
        Predicts the main conformations of the fragment.

        Predicts the main conformations by sampling from a probability distribution stored in self._model.

        Parameters
        ----------
        rng : np.random.default_rng()
            Default random number generator of numpy.

        Returns
        -------
        numpy.ndarray
            The predicted coordinates of the fragment.
        """
        return self._main_coords[self._model.predict(rng)]

