from artsm.model.models import ModelOneBead


class PredefMol:
    def __init__(self, atoms, masses, elements, confs, labels, p, d_max):
        """
        A class to represent predefined molecules. Currently, water, Na, and Cl.

        Parameters
        ----------
            atoms : numpy.ndarray
                Atom names.
            masses : numpy.ndarray
                Atom masses.
            elements : numpy.ndarray
                Atom elements.
            confs : numpy.ndarray
                The main conformations.
            labels : numpy.ndarray
                Clustering labels
            p : numpy.ndarray
                Probability of each clustering label.
            d_max : numpy.ndarray
                Maximum distance of the center of mass to any atom in each main conformation.
        """
        self.atoms = atoms
        self.atom_order = atoms
        self.n_atoms = atoms.size
        self.elements = elements
        self.masses = masses
        self._main_coords = confs
        self._model = ModelOneBead(labels, p)
        self.d_max = d_max

    def predict_confs(self, rng):
        """
        Sample a main conformation according to the probabilities p.
        Parameters
        ----------
        rng : np.random.default_rng()
            Default random number generator of numpy.
        Returns
        -------
            tuple
                Contains two values
                    numpy.ndarray
                        Coordinates of the predicted main conformation.
                    float
                        Maximum distance of the center of mass to any atom in the main conformation.
        """
        idx = self._model.predict(rng)
        return self._main_coords[idx], self.d_max[idx]


# Water and Ion classes are the same as PredefMol.
# However, if specific adjustments are necessary in the future, this implementation has the flexibility to easily do so.
class Water(PredefMol):
    def __init__(self, atoms, masses, elements, confs, labels, p, d_max):
        super().__init__(atoms, masses, elements, confs, labels, p, d_max)


class Ion(PredefMol):
    def __init__(self, atoms, masses, elements, confs, labels, p, d_max):
        super().__init__(atoms, masses, elements, confs, labels, p, d_max)


class OneToOne:
    def __init(self, name):
        self.name = name
        self.n_atoms = 1

