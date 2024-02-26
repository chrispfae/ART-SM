import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from artsm.model.models import ModelOneBead
import artsm.utils.other
from artsm.utils.fileparsing import write_yaml
from artsm.utils.plots import plot_bar


def _evaluate_scores(scores, mean_prob, path):
    """
    Write evaluation scores to a file and generate a bar plot.

    The following is written to the file:
    - Mean value of the labels (probabilities)
    - Mean and standard deviation of the scores
    - Scores

    Parameters
    ----------
    scores : numpy.ndarray)
        Array of evaluation scores.
    mean_prob : float
        Mean value of the labels (probabilities)
    path : str
        Output directory.
    """
    filename = os.path.join(path, 'cross_validation.txt')
    output_file = open(filename, 'w')
    output_file.write(f'Mean probability: {mean_prob}\n')
    output_file.write(f'Mean scores: {scores.mean()}\n')
    output_file.write(f'Std scores: {scores.std()}\n')
    output_file.write(f'Scores: {scores}\n')
    output_file.close()

    filename = os.path.join(path, 'scores.png')
    plot_bar(scores, filename, 'CV Scores', ylab='MAE')


def _model_performance(X, Y, path, rng, seed=None):
    """
    Calculate the model performance via 10-fold cross-validation and save results to file.

    Parameters
    ----------
    X : pandas.DataFrame
        The input dataset.
    Y : pandas.Series
        Labels
    path : str
        Output directory.
    rng : np.random.default_rng()
        Default random number generator of numpy.
    seed : int, default None
        Seed for reproducible results.
    """
    idx = rng.permutation(Y.size)
    X_train = X.reindex(idx).reset_index(drop=True)
    Y_train = Y.reindex(idx).reset_index(drop=True)
    mean_prob = np.mean(Y_train.to_numpy())

    rand_tree = RandomForestRegressor(random_state=seed)
    scores = cross_val_score(rand_tree, X_train, Y_train.to_numpy(), scoring='neg_mean_absolute_error', cv=10)
    mae_scores = -scores
    _evaluate_scores(mae_scores, mean_prob, path)


def training(X, Y, path, rng, seed=None):
    """
    Trains a RandomForestRegressor model on the given input data.

    Parameters
    ----------
    X : pandas.DataFrame
        The input dataset.
    Y : pandas.Series
        Labels
    path : str
        Output directory.
    rng : np.random.default_rng()
        Default random number generator of numpy.
    seed : int, default None
        Seed for reproducible results

    Returns
    -------
    The trained RandomForestRegressor model.
    """
    model = RandomForestRegressor(random_state=seed)
    model.fit(X, Y.to_numpy())
    _model_performance(X, Y, path, rng, seed)
    return model


def probabilities_one_bead(labels, path):
    """
    Calculate the frequency of each label and return a ModelOneBead object to be used for one bead molecules.

    Calculated probabilities are written to file.

    Parameters
    ----------
    labels : numpy.ndarray
    path : str
        Output directory.

    Returns
    -------
    ModelOneBead
        Model storing the labels corresponding frequencies.
    """
    # Determine model
    unique_elements, counts = np.unique(labels, return_counts=True)
    p = counts / np.sum(counts)
    model = ModelOneBead(unique_elements, p)

    # Write probabilities to file
    p_dict = dict(zip(unique_elements, p))
    p_dict_converted = {int(key): float(value) for key, value in p_dict.items()}
    filename = os.path.join(path, 'probabilities.yaml')
    write_yaml(filename, p_dict_converted)

    return model
