import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_bar(data, filename, title='Histogram', ylab='y'):
    """
    Plot basic histogram and save to file.

    Parameters
    ----------
    data : numpy.ndarray
        The height of each bar of the histogram.
    filename : str
    title : str, optional
    ylab : str, optional
    """
    x = np.arange(data.size)
    plt.bar(x, data)
    plt.title(title)
    plt.ylabel(ylab)
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_histogram(data, filename, title='Histogram', xlab='x', binning=50):
    """
    Plot basic histogram and save to file.

    Parameters
    ----------
    data : list or numpy.ndarray
        Data points to generate the histogram from.
    filename : str
    title : str, optional
    xlab : str, optional
    binning : int or list or numpy.ndarray
        Either the number of bins or directly the intervals for each bin.
    """
    plt.hist(data, bins=binning)
    plt.title(title)
    plt.xlabel(xlab)
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_scatter(x, y, filename, title="Scatter plot", xlab='x', ylab='y', point_size=10):
    """
    Create basic scatter plot and save to file.
    Parameters
    ----------
    x : list or numpy.ndarray
    y : list or numpy.ndarray
    filename : str
    title : str, optional
    xlab : str, optional
    ylab : str, optional
    point_size : int, optional
    """
    plt.scatter(x, y, s=point_size)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_dendrogram(data, filename, title='Dendrogram', ylab='y', threshold=0):
    """
    Create dendrogram and save to file.

    Parameters
    ----------
    data : numpy.ndarray
        Result of scipy.cluster.hierarchy.linkage
    filename : str
    title : str, optional
    ylab : str, optional
    threshold : int
        Cutoff for determining the number of clusters. Has an influence on the coloring of the individual data points.
    """

    dendrogram(data, no_labels=True, color_threshold=threshold)
    plt.title(title)
    plt.ylabel(ylab)
    plt.savefig(filename, dpi=300)
    plt.close()
