from libc.math cimport atan2, sqrt


def calc_angle(atom1, atom2, atom3):
    """
    Calculate the angle between three atoms.

    Parameters
    ----------
    atom1 : list
        The coordinates of the first atom.
    atom2 : list
        The coordinates of the second atom.
    atom3 : list
        The coordinates of the third atom.

    Returns
    -------
    angle : float
        The angle between the three atoms.
    """
    cdef double[3] atom1_c
    cdef double[3] atom2_c
    cdef double[3] atom3_c

    atom1_c[0] = atom1[0]
    atom1_c[1] = atom1[1]
    atom1_c[2] = atom1[2]
    atom2_c[0] = atom2[0]
    atom2_c[1] = atom2[1]
    atom2_c[2] = atom2[2]
    atom3_c[0] = atom3[0]
    atom3_c[1] = atom3[1]
    atom3_c[2] = atom3[2]
    return calc_angle_c(atom1_c, atom2_c, atom3_c)


cdef double calc_angle_c(double[3] atom1, double[3] atom2, double[3] atom3):
    """
    Calculate the angle between three atoms.

    Defined in C to speed up the calculation.

    Parameters
    ----------
    atom1 : list
        The coordinates of the first atom.  Declare as a C arr.
    atom2 : list
        The coordinates of the second atom. Declare as a C arr.
    atom3 : list
        The coordinates of the third atom. Declare as a C arr.

    Returns
    -------
    angle : float
        The angle between the three atoms.
    """
    cdef double[3] cp1
    cdef double[3] cp2
    cdef double[3] xp
    cdef double x, y

    cp1[0] = atom1[0] - atom2[0]
    cp1[1] = atom1[1] - atom2[1]
    cp1[2] = atom1[2] - atom2[2]
    cp2[0] = atom3[0] - atom2[0]
    cp2[1] = atom3[1] - atom2[1]
    cp2[2] = atom3[2] - atom2[2]

    x = cp1[0] * cp2[0] + cp1[1] * cp2[1] + cp1[2] * cp2[2]

    xp[0] = cp1[1] * cp2[2] - cp1[2] * cp2[1]
    xp[1] = -cp1[0] * cp2[2] + cp1[2] * cp2[0]
    xp[2] = cp1[0] * cp2[1] - cp1[1] * cp2[0]

    y = sqrt(xp[0]*xp[0] + xp[1]*xp[1] + xp[2]*xp[2])

    return atan2(y, x)


def calc_dihedral(atom1, atom2, atom3, atom4):
    """
    Calculate the dihedral angle between four atoms.

    Parameters
    ----------
    atom1 : list
        The coordinates of the first atom.
    atom2 : list
        The coordinates of the second atom.
    atom3 : list
        The coordinates of the third atom.
    atom4 : list
        The coordinates of the fourth atom.

    Returns
    -------
    angle : float
        The dihedral angle between the four atoms.
    """

    cdef double[3] atom1_c
    cdef double[3] atom2_c
    cdef double[3] atom3_c
    cdef double[3] atom4_c

    atom1_c[0] = atom1[0]
    atom1_c[1] = atom1[1]
    atom1_c[2] = atom1[2]
    atom2_c[0] = atom2[0]
    atom2_c[1] = atom2[1]
    atom2_c[2] = atom2[2]
    atom3_c[0] = atom3[0]
    atom3_c[1] = atom3[1]
    atom3_c[2] = atom3[2]
    atom4_c[0] = atom4[0]
    atom4_c[1] = atom4[1]
    atom4_c[2] = atom4[2]
    return calc_dihedral_c(atom1_c, atom2_c, atom3_c, atom4_c)

    
cdef double calc_dihedral_c(double[3] atom1, double[3] atom2, double[3] atom3, double[3] atom4):
    """
    Calculate the dihedral angle between four atoms.

    Defined as a C function to speed up the calculation.

    Parameters
    ----------
    atom1 : list
        The coordinates of the first atom. Declare as a C arr.
    atom2 : list
        The coordinates of the second atom. Declare as a C arr.
    atom3 : list
        The coordinates of the third atom. Declare as a C arr.
    atom4 : list
        The coordinates of the fourth atom. Declare as a C arr.

    Returns
    -------
    angle : float
        The dihedral angle between the four atoms.
    """
    
    cdef double v1[3]
    cdef double v2[3]
    cdef double v3[3]

    v1[0] = atom2[0] - atom1[0]
    v1[1] = atom2[1] - atom1[1]
    v1[2] = atom2[2] - atom1[2]

    v2[0] = atom3[0] - atom2[0]
    v2[1] = atom3[1] - atom2[1]
    v2[2] = atom3[2] - atom2[2]

    v3[0] = atom4[0] - atom3[0]
    v3[1] = atom4[1] - atom3[1]
    v3[2] = atom4[2] - atom3[2]

    cdef double n1[3]
    cdef double n2[3]
    cdef double cp1[3]
    cdef v2_norm
    cdef double x, y

    n1[0] = -v1[1] * v2[2] + v1[2] * v2[1]
    n1[1] = v1[0] * v2[2] - v1[2] * v2[0]
    n1[2] = -v1[0] * v2[1] + v1[1] * v2[0]

    n2[0] = -v2[1] * v3[2] + v2[2] * v3[1]
    n2[1] = v2[0] * v3[2] - v2[2] * v3[0]
    n2[2] = -v2[0] * v3[1] + v2[1] * v3[0]

    x = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2])

    cp1[0] = n1[1] * n2[2] - n1[2] * n2[1]
    cp1[1] = -n1[0] * n2[2] + n1[2] * n2[0]
    cp1[2] = n1[0] * n2[1] - n1[1] * n2[0]

    v2_norm = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])

    y = (cp1[0] * v2[0] + cp1[1] * v2[1] + cp1[2] * v2[2]) / v2_norm

    return atan2(y, x)
