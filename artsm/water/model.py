import sys

from artsm.model.models import ModelOneBead
from artsm.utils.other import setup_logger
from artsm.water.data import supported_water_models


def get_water_model(model='TIP3P'):
    """
    Return a Water object given its name.

    Parameters
    ----------
    model : str, default 'TIP3P'
        The name of the water model to retrieve.

    Returns
    -------
    Water

    Raises
    ------
    SystemExit
        If the specified water model is not available.
    """
    if model in supported_water_models:
        return Water(**supported_water_models[model])
    else:
        logger = setup_logger(__name__)
        logger.error(f'The specified Water model {model} is not available. '
                     f'However, the following water models are supported: {supported_water_models.keys()}')
        sys.exit(-1)


class Water:
    def __init__(self, atoms, masses, elements, confs, labels, p, d_max):
        """
        A class to represent a water model.

        Parameters
        ----------
            atoms : numpy.ndarray
                Atom names.
            masses : numpy.ndarray
                Atom masses.
            elements : numpy.ndarray
                Atom elements.
            confs : numpy.ndarray
                The main conformations of the water model.
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
