import os

from MDAnalysis.topology.guessers import guess_masses, guess_types
import numpy as np

from artsm.resolution_transformation.backmap import preproc, prediction
from artsm.model.features import com_distance, preprocessing
from artsm.model.training import training
from artsm.model.mainconfs import main_conformations
from artsm.utils.angles import calc_dihedral
from artsm.utils.containers import reorder_atom_group
from artsm.utils.fileparsing import write_smiles
from artsm.utils.other import mda_selection


class FragmentPair:
    def __init__(self, fr1, fr2, con, smiles=None):
        """
        Class representing a fragment pair.

        Parameters
        ----------
        fr1 : Fragment
            The first fragment.
        fr2 : Fragment
            The second fragment.
        con : Connector
            The connector between the first and second fragment.
        smiles : str, default None
            The SMILES string of the fragment pair.
        """
        self.fr1 = fr1
        self.fr2 = fr2
        self.con = con
        if smiles is not None:
            self.smiles = smiles

        # Data storing
        self._fr1_zmatrix_data = []
        self._fr1_coords_data = []
        self._fr2_zmatrix_data = []
        self._fr2_coords_data = []
        self._X = []
        self._dihedral_data = []

        # Models
        self._main_coords1 = None
        self._main_internal1 = None
        self._main_coords2 = None
        self._main_internal2 = None
        self._main_dihedrals = None
        self._model = None
        self.reverse = False

        # confs
        self.pred_dihedral = None

    def extract_fr_data(self, residue):
        """
        Extracts data for a given residue, including internal coordinates, center of mass distance between the two
        fragments, and the dihedral angle of the connector.

        Parameters
        residue : MDAnalysis.Residue
        """
        self._calc_zmatrix(residue)
        self._extract_features(residue)
        self._calc_dihedral_connector(residue)

    def _calc_zmatrix(self, residue):
        """
        Calculate the dihedral angles of the z-matrix for both fragments from the given residue and
        store them as attributes.

        The coordinates of the individual fragments are calculated and stores as well.

        Parameters
        ----------
            residue : MDAnalysis.Residue
        """
        fr1_zmatrix_data, fr1_coords_data = self.fr1.calc_zmatrix(residue)
        fr2_zmatrix_data, fr2_coords_data = self.fr2.calc_zmatrix(residue)
        self._fr1_zmatrix_data.append(fr1_zmatrix_data)
        self._fr1_coords_data.append(fr1_coords_data)
        self._fr2_zmatrix_data.append(fr2_zmatrix_data)
        self._fr2_coords_data.append(fr2_coords_data)

    def _extract_features(self, residue):
        """
        Extract features from the given residue from the given residue and store as attributes.

        Currently only the distance between the centers of mass of the two fragments is calculated.

        Parameters
        ----------
        residue: MDAnalysis.Residue
        """
        fr1_simulation = residue.atoms.select_atoms(mda_selection(self.fr1.atoms))
        fr2_simulation = residue.atoms.select_atoms(mda_selection(self.fr2.atoms))
        com = com_distance(fr1_simulation, fr2_simulation)
        data = np.array([com])
        self._X.append(data)

    def _calc_dihedral_connector(self, residue):
        """
        Calculate the dihedral angle of the connector from the given residue and store as attribute.

        Parameters
        ----------
        residue: MDAnalysis.Residue
        """
        atoms = residue.atoms.select_atoms(mda_selection(self.con.atoms))
        atoms_reordered = reorder_atom_group(atoms, self.con.atoms)
        coords = atoms_reordered.positions
        dihedral = calc_dihedral(coords[0], coords[1], coords[2], coords[3])
        dihedral = np.array([dihedral])
        self._dihedral_data.append(dihedral)

    def set_data(self, fr1_zmatrix_data, fr1_coords_data, fr2_zmatrix_data, fr2_coords_data, X, dihedral):
        """
        Set the data of the FragmentPair object.

        Parameters
        ----------
            fr1_zmatrix_data : list
                The z-matrix data of fragment 1.
            fr1_coords_data : list
                The coordinates of fragment 1.
            fr2_zmatrix_data : list
                The z-matrix data of fragment 2.
            fr2_coords_data : list
                The coordinates of fragment 2.
            X : list
                The features data. Currently the center of mass distance between the two fragments.
            dihedral :list
                The dihedral angles of the connector.
        """
        self._fr1_zmatrix_data = fr1_zmatrix_data
        self._fr1_coords_data = fr1_coords_data
        self._fr2_zmatrix_data = fr2_zmatrix_data
        self._fr2_coords_data = fr2_coords_data
        self._X = X
        self._dihedral_data = dihedral

    def get_data(self):
        """
        Returns the data associated with the fragment pair.

        Returns
        -------
            tuple: A tuple containing the following arrays:
                fr1_zmatrix_data : list
                    The z-matrix data of fragment 1.
                fr1_coords_data : list
                    The coordinates of fragment 1.
                fr2_zmatrix_data : list
                    The z-matrix data of fragment 2.
                fr2_coords_data : list
                    The coordinates of fragment 2.
                X : list
                    The features data. Currently the center of mass distance between the two fragments.
                dihedral_data :list
                    The dihedral angles of the connector.
        """
        return np.array(self._fr1_zmatrix_data), np.array(self._fr1_coords_data), \
               np.array(self._fr2_zmatrix_data), np.array(self._fr2_coords_data), \
               np.array(self._X), np.array(self._dihedral_data)

    def set_models(self, main_coords1, main_internal1, main_coords2, main_internal2, dihedrals, model):
        """
        Set the models of the FragmentPair object.

        Parameters
        ----------
        main_coords1 : numpy.ndarray
            The coordinates of the main conformations of fragment 1.
        main_internal1 : numpy.ndarray
            The internal coordinates of the main conformations of fragment 1.
        main_coords2 : numpy.ndarray
            The coordinates of the main conformations of fragment 2.
        main_internal2 : numpy.ndarray
            The internal coordinates of the main conformations of fragment 2.
        dihedrals : numpy.ndarray
            The dihedral angles of the main conformations of the connector.
        model : Model
            Machine learning model.
        """
        self._main_coords1 = main_coords1
        self._main_internal1 = main_internal1
        self._main_coords2 = main_coords2
        self._main_internal2 = main_internal2
        self._main_dihedrals = dihedrals
        self._model = model

    def get_models(self):
        """
        Returns the models associated with the fragment pair.

        Returns
        -------
        main_coords1 : numpy.ndarray
            The coordinates of the main conformations of fragment 1.
        main_internal1 : numpy.ndarray
            The internal coordinates of the main conformations of fragment 1.
        main_coords2 : numpy.ndarray
            The coordinates of the main conformations of fragment 2.
        main_internal2 : numpy.ndarray
            The internal coordinates of the main conformations of fragment 2.
        main_dihedrals : numpy.ndarray
            The dihedral angles of the main conformations of the connector.
        model : Model
            Machine learning model.
         """
        return self._main_coords1, self._main_internal1, self._main_coords2, \
               self._main_internal2, self._main_dihedrals, self._model

    def derive_models(self, path, rng, seed=None):
        """
        Derive models, i.e. determine the main conformations of both fragments and the connector
        via hierarchical clustering, and train the machine learning model.

        Results are stored as attributes. Figures are created to verify the clustering and training process.

        Parameters
        ----------
        path : str
            Output directory.
        rng : np.random.default_rng()
            Default random number generator of numpy.
        seed : int, default None
            Seed for reproducible results.
        """
        write_smiles(os.path.join(path, 'smiles.txt'), self.smiles)

        fr1_path = os.path.join(path, 'mainconfs1')
        write_smiles(os.path.join(fr1_path, 'smiles.txt'), self.fr1.smiles)
        labels1, representative_idx1 = main_conformations(self._fr1_zmatrix_data, self.fr1.zmatrix, fr1_path)
        self._main_coords1 = self._fr1_coords_data[representative_idx1]
        self._main_internal1 = self._fr1_zmatrix_data[representative_idx1]

        fr2_path = os.path.join(path, 'mainconfs2')
        write_smiles(os.path.join(fr2_path, 'smiles.txt'), self.fr1.smiles)
        labels2, representative_idx2 = main_conformations(self._fr2_zmatrix_data, self.fr2.zmatrix, fr2_path)
        self._main_coords2 = self._fr2_coords_data[representative_idx2]
        self._main_internal2 = self._fr2_zmatrix_data[representative_idx2]

        dihedral_path = os.path.join(path, 'dihedral')
        write_smiles(os.path.join(dihedral_path, 'smiles.txt'), "".join(self.con.elements))
        labels_dihedral, representative_dihedral = main_conformations(self._dihedral_data,
                                                                      [self.con.elements], dihedral_path)
        self._main_dihedrals = self._dihedral_data[representative_dihedral]

        training_path = os.path.join(path, 'ml')
        X, Y = preprocessing(labels1, labels2, labels_dihedral, self._X)
        self._model = training(X, Y, training_path, rng, seed)

    def predict_confs(self, residue, rng, pred_internal1=None):
        """
        Predicts main conformations of both fragments and the connector given a coarse grained residue.

        Parameters
        ----------
        residue: MDAnalysis.Residue
        rng : np.random.default_rng()
            Default random number generator of numpy.
        pred_internal1 : default None
            Everything but None indicates that the conformation of the first fragment
            has already been determined in a previous step.

        Returns
        -------
        tuple of numpy.ndarray
            Contains coordinates and internal coordinates of the fragments and the connector.

        """
        X = preproc(self, residue, pred_internal1)
        return prediction(self, X, residue, rng)


class Connector:
    def __init__(self, atoms, bond_types):
        """
        Class representing a connector, which always consists of four atoms.

        Parameters
        ----------
            atoms : numpy.ndarray
                Atom names.
            bond_types : numpy.ndarray
                The bond types between the atoms. Three values. 1 indicates single bond, 2 double bond, ...
        """
        self.atoms = atoms
        self.n_atoms = self.atoms.size
        self.elements = guess_types(atoms)
        self.masses = guess_masses(self.elements)
        self.bond_types = bond_types
        self.bond = (self.elements[1], self.elements[2], bond_types[1])
        self.angle1 = (self.elements[0], self.elements[1], self.elements[2], bond_types[0], bond_types[1])
        self.angle2 = (self.elements[1], self.elements[2], self.elements[3], bond_types[1], bond_types[2])
