import numpy as np
import pandas as pd


def labeling(X, missing_combinations):
    """
    Assigns labels (probabilities) to the input data based on the frequency of each combination of main conformations.

    Parameters
    ----------
    X : pandas.DataFrame
        Input data extracted from atomistic simulations
        Index: Range index
        Columns: Fragment1 Fragment2 Connector COM-distance
    missing_combinations : pandas.DataFrame
        Combinations of main conformations that do not occur in the atomistic simulations. Receive probability zero.

    Returns
    -------
    numpy.ndarray
        Labels assigned to each observation in X, including zero probabilities for missing combinations.
    """
    binning, prob = np.unique(X, axis=0, return_counts=True)
    prob = prob / prob.size
    y = np.zeros(X.shape[0])
    for i in range(binning.shape[0]):
        pos = np.where((X == binning[i]).all(axis=1))[0]
        y[pos] = prob[i]
    # Add zero probabilities
    zero_prob = np.zeros(missing_combinations.shape[0])
    y_zeroprob = np.concatenate((y, zero_prob))

    return y_zeroprob
