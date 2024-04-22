import os

import numpy as np


def _prepare_fr_pair_folders(path, idx):
    """
    Create directories to save figures of hierarchical clustering for fragment pairs.

    Creates folders 'mainconfs1', 'mainconfs2', 'dihedral' and 'ml' for each given index.

    Parameters
    ----------
    path : str
        Output directory
    idx : list
        Fragment pair indices.
    """
    path = os.path.join(path, 'build_db')
    os.makedirs(path, exist_ok=True)
    for i in idx:
        fr_pair_path = os.path.join(path, f'fr_pair{i}')
        os.makedirs(fr_pair_path, exist_ok=True)
        os.makedirs(os.path.join(fr_pair_path, 'mainconfs1'), exist_ok=True)
        os.makedirs(os.path.join(fr_pair_path, 'mainconfs2'), exist_ok=True)
        os.makedirs(os.path.join(fr_pair_path, 'dihedral'), exist_ok=True)
        os.makedirs(os.path.join(fr_pair_path, 'ml'), exist_ok=True)


def _prepare_fr_folders(path, idx):
    """
    Create directories to save figures of hierarchical clustering for fragments.

    Creates folders 'mainconfs' and 'ml' for each given index.

    Parameters
    ----------
    path : str
        Output directory
    idx : list
        Fragment indices.
    """
    path = os.path.join(path, 'build_db')
    os.makedirs(path, exist_ok=True)
    for i in idx:
        fr_path = os.path.join(path, f'fragment{i}')
        os.makedirs(fr_path, exist_ok=True)
        os.makedirs(os.path.join(fr_path, 'mainconfs'), exist_ok=True)
        os.makedirs(os.path.join(fr_path, 'ml'), exist_ok=True)


def derive_models(database, path, rng, seed=None, ignore_fr_pairs=None, ignore_fr=None):
    """
    Derives models (main conformations and ML models/probabilities) for one bead molecules and fragment pairs
    in the given database.

    Parameters
    ----------
    database : Database
        The database object containing fragment pairs and fragments.
    path : str
        Output directory.
    rng : np.random.default_rng()
        Default random number generator of numpy.
    seed : int, default None
        Seed for reproducible results
    ignore_fr_pairs : list, default None
        A list of fr_pair indices to ignore.
    ignore_fr : list, default None
        A list of fragment indices to ignore.
    """

    # Fragment pair models
    if ignore_fr_pairs is None:
        fr_pair_ids = database.get_fr_pair_ids()
    else:
        fr_pair_ids = list(set(database.get_fr_pair_ids() or []) - set(ignore_fr_pairs))
    if fr_pair_ids:
        _prepare_fr_pair_folders(path, fr_pair_ids)
        for fr_pair_id in fr_pair_ids:
            fr_pair = database.get_fr_pair(fr_pair_id, rng)
            path_fr_pair = os.path.join(path, f'build_db/fr_pair{fr_pair_id}')
            fr_pair.derive_models(path_fr_pair, rng, seed)
            database.update_fr_pair_models(fr_pair_id, fr_pair.get_models())

    # One bead models
    if ignore_fr is None:
        fr_ids = database.get_fr_ids(data_required=True)
    else:
        fr_ids = list(set(database.get_fr_ids(data_required=True) or []) - set(ignore_fr))

    if fr_ids:
        _prepare_fr_folders(path, fr_ids)
        for fr_id in fr_ids:
            fr = database.get_fr(fr_id, rng)
            path_fr = os.path.join(path, f'build_db/fragment{fr_id}')
            fr.derive_models(path_fr)
            database.update_fr_models(fr_id, fr.get_models())


class ModelOneBead:
    def __init__(self, labels, p):
        """
        Model for one bead molecules.

        The model is defined by a set of labels and a probability distribution over those labels.

        Parameters
        ----------
        labels : list
            Cluster labels.
        p : list
            Probability of each label.
        """
        self.labels = labels
        self.p = p

    def predict(self, rng):
        """Return a label according to the probability distribution."""
        return rng.choice(a=self.labels, size=1, p=self.p)[0]
