import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import MDAnalysis as mda

import re

from artsm.utils.angles import calc_dihedral

x = np.random.normal(4, 5, 15).reshape((5, 3))
print(x)
print(calc_dihedral(x[0], x[1], x[2], x[3]))
print(calc_dihedral(x[0], x[1], x[2], x[4]))
x[2:, :] = x[2:, :] + np.array([[2, 4, 1],
                                [2, 4, 1],
                                [2, 4, 1]])
print(x)
print(calc_dihedral(x[0], x[1], x[2], x[3]))
print(calc_dihedral(x[0], x[1], x[2], x[4]))



mapping_dict = {1: {'C': 'C1', 'CA': 'CA1', 'O': 'O1', 'CB': 'CB1', 'N': 'N1', 'HA': 'HA3', 'HB1': 'HB1',
                      'HB2': 'HB2', 'HB3': 'HB3', 'H1': 'HN1', 'H2': 'HN2'},
                2: {'C': 'C2', 'CA': 'CA2', 'O': 'O2', 'CB': 'CB2', 'N': 'N2', 'HA': 'HA4', 'HB1': 'HB4',
                      'HB2': 'HB5', 'HB3': 'HB6', 'HN': 'HN3'},
                0: {'OT1': 'O3', 'OT2': 'O4', 'C': 'C3', 'CA': 'CA3', 'N': 'N3', 'HT2': 'HO1', 'HA1': 'HA1',
                      'HA2': 'HA2', 'HN': 'H1'}}
offset = 0
previous = ''
running = 0
outfile = open('/home/chris/temp/workstation/atomistic_all/backward_mod.gro', 'w')
filename = '/home/chris/temp/workstation/atomistic_all/0-backward_old_naming.gro'
with open(filename) as infile:
    for i, line in enumerate(infile):
        ala = re.search(r'ALA', line)
        gly = re.search(r'GLY', line)
        if ala is None and gly is None:
            offset += 1
            outfile.write(line)
        else:
            string = line[:15]
            # Replace id
            id_ = int(re.search(r'\s+(\d+)', string).group(1))
            if previous != id_ and id_ % 3 == 1:
                running += 1
            id_str = f'{running}'
            if len(id_str) > 5:
                print('Too many residues. Bail out ...')
                sys.exit(-1)
            new_id = ' ' * (5 - len(id_str)) + id_str
            # Replace residue name
            new_resid = re.sub(r'GLY', 'AAG', re.sub(r'ALA', 'AAG', string))
            # Replace atom name
            dict_ = mapping_dict[id_ % 3]
            match = re.search(r'\s+\d+AAG\s+([\d\w]+)', new_resid)
            atom_name = match.group(1)
            start, stop = match.regs[1]
            new_atom_name = dict_[atom_name]
            offset = len(new_atom_name) - len(atom_name)
            new_line = new_id + new_resid[5:start - offset] + new_atom_name + line[15:]
            outfile.write(new_line)
            previous = id_



outfile.close()






def preprocess(filename):
    data = np.load(filename)
    data[data < 0] += 360
    return data.reshape(data.size, 1)


atomistic = preprocess('/home/chris/temp/workstation/kde/atomistic_O_C1_C2_C3_values.npy')
backmap = preprocess('/home/chris/temp/workstation/kde/backmap_O_C1_C2_C3_values.npy')
posre = preprocess('/home/chris/temp/workstation/kde/posre_O_C1_C2_C3_values.npy')

kde_atomistic = KernelDensity(kernel='gaussian', bandwidth=5).fit(atomistic)
kde_posre = KernelDensity(kernel='gaussian', bandwidth=5).fit(posre)

x = np.arange(0, 361, 1)
x.shape = (x.size, 1)

y_atomistic = np.exp(kde_atomistic.score_samples(x))
y_posre = np.exp(kde_posre.score_samples(x))

plt.fill_between(x.flatten(), y_atomistic, 0, color='tab:gray', label='atomistic', alpha=0.4, linewidth=5)
plt.plot(x, y_posre, color='tab:orange', label='Position restraint')
# plt.hist(atomistic, bins=72, density=True, alpha=0.8)
# plt.hist(posre, bins=72, density=True, alpha=0.8)
plt.legend()
plt.show()
