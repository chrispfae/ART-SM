import os

from MDAnalysis.topology.guessers import guess_types
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from artsm.utils.other import setup_logger
from artsm.utils.plots import plot_dendrogram, plot_histogram, plot_scatter


def _plot_dihedrals(data, zmatrix, path):
    """
    Plot histograms of internal coordinates (dihedral angles) and save to file.

    Parameters
    ----------
    data : numpy.ndarray
        Internal coordinates.
    zmatrix : list of numpy.ndarray
        Atom names of each dihedral angle.
    path : str
        Output directory.
    """
    assert data.shape[1] == len(zmatrix)
    for i in range(data.shape[1]):
        filename = os.path.join(path, f'dihedral_number{i}_elements{"_".join(guess_types(zmatrix[i]))}.png')
        plot_histogram(data[:, i], filename, title='Dihedral angle histogram', xlab='Dihedral angle')


def _plot_clustering(linkage_order, linkage_criteria, threshold, path):
    """
    Plot the result of hierarchical clustering.

    Parameters
    ----------
    linkage_order : numpy.ndarray
        Return of scipy.cluster.hierarchy.linkage function.
        columns are [i j k l]. Cluster i is connected to cluster j with d(i, j) = k.
        l is number of observations of the newly formed cluster.
    linkage_criteria : str
    threshold : float
        Distance threshold for dendrogram coloring.
    path : str
        Output directory.
    """
    x = np.arange(0, len(linkage_order))
    y = linkage_order[:, 2]
    filename = os.path.join(path, 'cluster_distances.png')
    plot_scatter(x, y, filename, title='Cluster distances ordered', xlab='Merging operation', ylab='Cluster distance')

    filename = os.path.join(path, 'hierarchical_clustering.png')
    title = f'Dendrogram molecules: {linkage_criteria} linkage'
    plot_dendrogram(linkage_order, filename, title=title, ylab='Distance', threshold=threshold)


def metric_dihedral_angle(angles1, angles2):
    """
    Calculate the distance between two arrays of angles.

    The metric angle is defined as the sum of the absolute differences between the individual angles.
    Periodicity of angles is considered.

    Parameters
    ----------
    angles1 : numpy.ndarray
    angles2 : numpy.ndarray

    Returns
    -------
    float:
        Angle metric
    """
    assert angles1.size == angles2.size
    assert np.all(angles1 >= -np.pi) and np.all(angles1 <= np.pi)
    assert np.all(angles2 >= -np.pi) and np.all(angles2 <= np.pi)
    diff = np.abs(angles1 - angles2)
    diff = np.minimum(diff, 2 * np.pi - diff)
    return np.sum(diff)


def _number_clusters(linkage_order):
    """
    Determines the number of clusters based on the linkage order.
    Parameters
    ----------
    linkage_order : np.ndarray
        Return of scipy.cluster.hierarchy.linkage function.
        columns are [i j k l]. Cluster i is connected to cluster j with d(i, j) = k.
        l is number of observations of the newly formed cluster.
    Returns
    -------
    int
        Number of clusters.
    """
    distance_norm = linkage_order[:, 2] / np.max(linkage_order[:, 2])
    distance_differences = distance_norm[1:] - distance_norm[:-1]
    jumps = np.where(distance_differences > 0.2)[0]
    if jumps.size == 0:
        number_clusters = 1
    else:
        number_clusters = distance_differences.size - np.min(jumps) + 1
    return number_clusters


def _dendrogram_threshold(linkage_order, number_clusters):
    """
    Determine the dendrogram threshold to obtain the specified number of clusters.
    Parameters
    ----------
    linkage_order : np.ndarray
        Return of scipy.cluster.hierarchy.linkage function.
        columns are [i j k l]. Cluster i is connected to cluster j with d(i, j) = k.
        l is number of observations of the newly formed cluster.
    number_clusters : int
        The desired number of clusters
    Returns
    -------
    float
        Distance threshold.
    """
    if number_clusters > 1:
        threshold = (linkage_order[-number_clusters, 2] + linkage_order[-(number_clusters - 1), 2]) / 2
    else:
        threshold = np.max(linkage_order[:, 2]) + 1.
    return threshold


def _hierarchical_clustering(data, linkage_criteria, path, clusters=None):
    """Performs hierarchical clustering and returns the label for each data point.
    Parameters
    ----------
    data: np.ndarray
        2D data on which clustering is performed.
    linkage_criteria: str
        Same as method in scipy.cluster.hierarchy.linkage
    path: str
        Output path for figures.
    clusters: int
        Number of different clusters. Automatically derived if not given.
    Returns
    -------
    np.ndarray
        Labels for each data point.
    """
    linkage_order = linkage(data, linkage_criteria, metric=metric_dihedral_angle)

    if clusters is None:
        clusters = _number_clusters(linkage_order)

    threshold = _dendrogram_threshold(linkage_order, clusters)

    _plot_clustering(linkage_order, linkage_criteria, threshold, path)

    labels = fcluster(linkage_order, t=threshold, criterion='distance') - 1
    return labels


def _representative_structure(data, labels):
    """
    Determine for each cluster the datapoint closest to the cluster mean and return its internal coordinates.

    Parameters
    ----------
    data : numpy.ndarray
        Internal coordinates.
    labels : numpy.ndarray
        Cluster label for each datapoint.

    Returns
    -------
    numpy.ndarray
        Internal coordinates of the cluster representative.
    """
    clusters = np.max(labels) + 1
    representatives = np.zeros(clusters, dtype=np.int_)
    for i in range(clusters):
        confs = data[labels == i, :]
        D = squareform(pdist(confs, metric=metric_dihedral_angle))
        row_sum = np.sum(D, axis=1)
        mediod = np.argmin(row_sum)
        representative_value = confs[mediod]
        representatives[i] = np.where(data == representative_value)[0][0]
    return representatives


def _check_main_conformations(data, labels, representatives, path):
    """
    Plot the result of hierarchical clustering for visual inspection.

    For 1D data a histogram is plotted. Each datapoint is colored according to its cluster assignment and
    representatives are visualized as vertical red lines.
    For 2D data a scatter plot is created. Each datapoint is colord according to its cluster assignment and
    representatives are visualized as large red dots.
    For higher dimensional data visualization is not implemented and a warning is logged.

    Parameters
    ----------
    data :  numpy.ndarray
        Internal coordinates.
    labels : numpy.ndarray
        Cluster label for each datapoint.
    representatives : numpy.ndarray
        Indices of the representative datapoints in the data array.
    path : str
        Output directory.
    """
    if data.shape[1] > 2:
        logger = setup_logger(__name__)
        logger.warning('too many dimensions to graphically check representatives.')
        return
    fig = plt.figure()
    fig.add_subplot(111, title='check clustering')
    if data.shape[1] == 1:
        for i in range(np.max(labels) + 1):
            plt.hist(data[labels == i], bins=np.arange(-np.pi, np.pi, (1/36)*np.pi))
            plt.axvline(data[representatives[i]], color='red')
    else:
        for i in range(np.max(labels) + 1):
            plt.scatter(data[labels == i, 0], data[labels == i, 1], s=1)
            plt.scatter(data[representatives[i], 0], data[representatives[i], 1], color='red', s=10)
    # plt.gca().set_aspect('equal', adjustable='box')
    filename = os.path.join(path, 'check_clustering.png')
    plt.savefig(filename, dpi=300)
    plt.close()


def main_conformations(data, zmatrix, path):
    """
    Determine the main conformations for a fragment or connector via hierarchical clustering.

    Plots of the clustering process are created for visual inspection.

    Parameters
    ----------
        data : numpy.ndarray
            Internal coordinates.
        zmatrix : list of numpy.ndarray
            Atom names of each dihedral angle.
        path : str
            Output directory
    Returns
    -------
        tuple
            Contains two values
                numpy.ndarray
                    Cluster label for each datapoint.
                numpy.ndarray
                    Indices of the representative datapoints of each cluster in the data array.
    """
    _plot_dihedrals(data, zmatrix, path)
    labels = _hierarchical_clustering(data, 'average', path)
    representative_idx = _representative_structure(data, labels)
    _check_main_conformations(data, labels, representative_idx, path)
    return labels, representative_idx
