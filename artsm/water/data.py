import numpy as np

from MDAnalysis.topology.guessers import guess_masses

from artsm.utils.other import center_of_mass


def sphere_radius(coords, com):
    """
    Calculate the maximum distance of the com to any given coordinate.

    The calculated distance is the radius of a sphere that is centered at the center of mass (com) of a set of
    coordinates (coords) and that contains all coordinates.

    Parameters
    ----------
    coords : numpy.ndarray
        2D array of coordinates.
    com : numpy.ndarray
        Center of mass of the coordinates. 1D array.

    Returns
    -------
    float
        Maximum distance of the com to any given coordinate which is the sphere radius.
    """
    distance = np.linalg.norm(coords - com, axis=1)
    return np.max(distance)


# TIP3P
atoms = np.array(['OH2', 'H1', 'H2', 'OH2', 'H1', 'H2', 'OH2', 'H1', 'H2', 'OH2', 'H1', 'H2'])
elements = np.array(['O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H'])
masses = guess_masses(elements)

p1 = 0.2428
p2 = 0.0639
p3 = 0.1934
p4 = 0.2186
p5 = 0.0633
p6 = 0.2180
p = np.array([p1, p2, p3, p4, p5, p6])

conf1 = np.array(
    [
        [6.670, 12.510, 21.920],
        [5.820, 12.230, 22.270],
        [7.300, 12.280, 22.610],
        [9.150, 15.940, 22.740],
        [9.800, 15.240, 22.780],
        [9.550, 16.650, 23.240],
        [7.340, 17.030, 19.370],
        [7.570, 16.180, 19.720],
        [7.670, 17.020, 18.470],
        [8.000, 14.570, 20.600],
        [7.910, 13.720, 21.030],
        [8.450, 15.110, 21.250],
    ]
)

conf2 = np.array(
    [
        [3.040, 29.030, 5.240],
        [2.470, 29.790, 5.320],
        [3.620, 29.240, 4.510],
        [0.830, 28.780, 2.420],
        [1.250, 28.700, 1.560],
        [1.280, 29.510, 2.840],
        [1.990, 31.320, 3.730],
        [1.190, 31.640, 4.140],
        [2.690, 31.830, 4.130],
        [-0.240, 27.350, 4.430],
        [0.430, 26.670, 4.310],
        [-0.080, 27.970, 3.720],
    ]
)

conf3 = np.array(
    [
        [38.250, 19.030, 14.510],
        [38.220, 18.400, 13.800],
        [38.700, 18.580, 15.220],
        [35.910, 20.090, 13.240],
        [36.760, 20.130, 13.680],
        [36.130, 19.840, 12.340],
        [34.720, 22.630, 12.720],
        [35.110, 21.750, 12.810],
        [34.790, 23.010, 13.600],
        [36.650, 19.550, 10.710],
        [37.390, 19.360, 10.130],
        [36.090, 20.140, 10.200],
    ]
)

conf4 = np.array(
    [
        [16.510, 21.770, 7.480],
        [17.200, 22.240, 7.940],
        [16.880, 21.570, 6.620],
        [20.490, 21.590, 4.290],
        [20.090, 22.430, 4.070],
        [20.460, 21.090, 3.470],
        [19.180, 21.910, 6.740],
        [19.220, 21.810, 5.790],
        [19.510, 22.800, 6.900],
        [19.960, 24.580, 7.070],
        [19.810, 24.820, 7.990],
        [19.850, 25.390, 6.580],
    ]
)

conf5 = np.array(
    [
        [27.960, 18.060, 38.690],
        [27.870, 17.930, 37.750],
        [27.280, 17.490, 39.070],
        [27.900, 18.350, 36.000],
        [27.890, 18.260, 35.040],
        [27.150, 18.920, 36.190],
        [26.120, 16.000, 39.320],
        [26.770, 15.310, 39.440],
        [25.770, 15.860, 38.440],
        [25.760, 15.930, 36.530],
        [26.630, 16.010, 36.140],
        [25.330, 15.230, 36.030],
    ]
)

conf6 = np.array(
    [
        [25.020, 15.760, 35.260],
        [24.420, 15.870, 34.510],
        [25.720, 16.390, 35.100],
        [24.520, 12.970, 35.160],
        [25.090, 12.300, 34.800],
        [25.110, 13.680, 35.410],
        [27.220, 17.360, 34.740],
        [27.360, 18.180, 35.230],
        [28.100, 17.070, 34.510],
        [23.370, 16.430, 33.110],
        [23.060, 15.740, 32.520],
        [22.870, 17.210, 32.860],
    ]
)

confs = np.array([conf1, conf2, conf3, conf4, conf5, conf6])

d_max = np.array([sphere_radius(coords, center_of_mass(coords, masses)) for coords in confs])

TIP3P = {'atoms': atoms, 'elements': elements, 'masses': masses,
         'confs': confs, 'labels': np.arange(0, confs.shape[0]), 'p': p, 'd_max': d_max}

supported_water_models = {'TIP3P': TIP3P}
