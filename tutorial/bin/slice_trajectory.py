import sys

import MDAnalysis as mda


pdb = sys.argv[1]
xtc = sys.argv[2]
u = mda.Universe(pdb, xtc)

count = 0
for ts in u.trajectory[::500]:
    count += 1

print(f'Number time steps: {count}')

u.atoms.write(sys.argv[3], frames=u.trajectory[::500])
