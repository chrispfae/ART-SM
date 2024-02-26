import sys
import MDAnalysis as mda

pdb = sys.argv[1]
xtc = sys.argv[2]
out = sys.argv[3]

u = mda.Universe(pdb, xtc)

u.trajectory[894]

u.atoms.write(out)
