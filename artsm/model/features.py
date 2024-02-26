import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from artsm.model.label import labeling


def com_distance(fr1, fr2):
    """
    Return the center of mass distance between two fragments.

    Parameters
    ----------
    fr1 : Fragment
    fr2 : Fragment

    Returns
    -------
    float
    """
    com_difference = fr1.center_of_mass() - fr2.center_of_mass()
    return np.linalg.norm(com_difference)


def _onehot(X, idx):
    """
    Apply one-hot encoding to selected columns in a DataFrame.

    Parameters
    ----------
    X : pandas.DataFrame
    idx : list
        The indices of the columns to be one-hot encoded.

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame with one-hot encoded columns.
    """
    idx = sorted(idx, reverse=True)
    X_mod = X.copy()
    for column in X_mod.columns[idx]:
        one_hot = pd.get_dummies(X[column])
        naming = list(one_hot.columns)
        naming = ['{}_{}'.format(column, i) for i in naming]
        one_hot.columns = naming
        X_mod = X_mod.drop(column, axis=1)
        X_mod = one_hot.join(X_mod)
    return X_mod


def _bin_feature(feature, bins=50, encode='ordinal', strategy='uniform'):
    """
    Bin a numerical feature into discrete intervals.

    Parameters
    ----------
    feature : pandas.DataFrame
        The numerical feature to be binned.
    bins : int, default 50
        The number of bins.
    encode : str, default 'ordinal'
        The encoding scheme for the binned feature.
        Can be 'ordinal', 'onehot-dense' or 'onehot'.
    strategy :str, default 'uniform'
        The strategy used to define the widths of the bins.
        Can be 'uniform', 'quantile', or 'kmeans'.

    Returns
    -------
    tuple of numpy.ndarray
        The binned feature and the mean values of the binned intervals.
    """
    model = KBinsDiscretizer(n_bins=bins, encode=encode, strategy=strategy, subsample=200_000)
    binned_feature = model.fit_transform(feature)
    binned_feature = binned_feature.astype(np.uint32)
    edges = model.bin_edges_[0]
    edges_mean = (edges[:-1] + edges[1:]) / 2.
    return binned_feature, edges_mean


def _bin_features(X, idx):
    """
    Bin the specified columns of the input DataFrame.

    Calls function ~artsm.model.features._bin_feature.

    Parameters
    ----------
    X : pandas.DataFrame
    idx : list
        The indices of the columns to binned.

    Returns
    -------
    tuple
        Contains two values:
            - pandas.DataFrame
                The modified DataFrame with binned columns.
            - list
                The mean values of the binned intervals.
    """
    X_mod = X.copy()
    edges = []
    for column in X.columns[idx]:
        X_mod[column], edge = _bin_feature(X[[column]])
        edges.append(edge)
    return X_mod, edges


def _stack_X(*data):
    """
    Stack the input data horizontally.

    Parameters
    ----------
        *data: tuple of 1D numpy.ndarray

    Returns
    -------
    np.ndarray
        2D horizontally stacked array.
    """
    for i in data:
        if len(i.shape) == 1:
            i.shape = (i.size, 1)
    return np.hstack(data)


def preprocessing(labels1, labels2, labels_dihedral, X_simulation):
    """
    Combine the features of the fragment pair (labels1, labels2, labels_dihedral) and the additional features
    (X_simulation) and preprocess them.

    The following steps are performed:
        1. Features are combined into a pandas.DataFrame.
        2. Numerical features are binned.
        3. Missing combinations are determined. Each datapoint is assigned a probability in the labeling step (5.).
           To this end, combinations of features that do not occur obtain a probability of zero.
        4. The fragment pair features are one hot encoded.
        5. Datapoints are labeled, i.e. each datapoints obtains a probability according to the simulation data.

    Parameters
    ----------
    labels1 : numpy.ndarray
        Main conformation label of the first fragment.
    labels2 : numpy.ndarray
        Main conformation label of the second fragment.
    labels_dihedral : numpy.ndarray
        Main conformation label of the connector.
    X_simulation : numpy.ndarray
        Additional features. Currently, the COM distance.

    Returns
    -------
    tuple
        Contains two values:
            - pandas.DataFrame
                The preprocessed dataset with one hot encoded features and missing combinations.
            - pandas.Series
                Labels (probabilities).

    """
    # Generate data frame - Split necessary to maintain data types
    data = _stack_X(labels1, labels2, labels_dihedral)
    X = pd.DataFrame(data, columns=['fr1', 'fr2', 'con'])
    X['comD'] = X_simulation

    X_binned, edges = _bin_features(X, [X.shape[1] - 1])

    # Determine missing combinations for zero probability
    n_mainconfs = np.array([np.max(labels1) + 1, np.max(labels2) + 1, np.max(labels_dihedral) + 1])
    missing_combinations = _find_missing_combinations(X_binned, ['fr1', 'fr2', 'con'], n_mainconfs)
    for i, feature in enumerate(missing_combinations.columns[3:]):
        missing_combinations[feature] = edges[i][[missing_combinations[feature].to_numpy()]].reshape(-1, 1)
    missing_combinations_onehot = _onehot(missing_combinations, [0, 1, 2])

    # labeling
    X_onehot = _onehot(X, [0, 1, 2])
    Y = labeling(X_binned, missing_combinations_onehot)
    Y = pd.Series(Y)

    # append zero prob
    X_zeroprob = pd.concat([X_onehot, missing_combinations_onehot], ignore_index=True)
    return X_zeroprob, Y


def _find_missing_combinations(X, features, n_classes):
    """
    Find missing combinations of features in the given dataset.

    First, all possible combinations of categorical features are determined, e.g. feature1 has 3 classes and
    feature2 has 2 classes -> Possible combinations 1-1 1-2 2-1 2-2 3-1 3-2. 
    The combinations not present in the given dataset are returned.

    Parameters
    ----------
    X : pandas.DataFrame
        Input dataset to determine the missing combinations.
    features : list of str
        Subset of features used to determine the missing combinations.
    n_classes : list
        The number of classes for each feature.

    Returns
    -------
    pd.DataFrame
        Missing combinations of features.
    """
    result = []

    unique_comD = X.iloc[:, -1].unique()

    for comD in unique_comD:
        existing_combinations = X.loc[X.iloc[:, -1] == comD, features].drop_duplicates()

        combinations = np.array(np.meshgrid(*[np.arange(labels) for labels in n_classes])).T.reshape(-1,
                                                                                                     len(n_classes))
        combinations_df = pd.DataFrame(combinations, columns=features)

        missing_combinations = pd.merge(combinations_df, existing_combinations, on=features, how='left', indicator=True)
        missing_combinations = missing_combinations.loc[missing_combinations['_merge'] == 'left_only', features]
        missing_combinations['comD'] = comD
        missing_combinations['comD'] = missing_combinations['comD'].astype('int64')

        result.extend(missing_combinations.values)

    return pd.DataFrame(result, columns=X.columns)
