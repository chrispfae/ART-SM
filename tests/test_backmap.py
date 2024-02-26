import filecmp
import sys
import warnings

import MDAnalysis as mda
import numpy as np

from artsm.build_db import main as build_db_main
from artsm.artsm import main as backmap_main
from fixtures import args_backmap_module, file_db0, file_db1


def test_seed(args_backmap_module):
    outdir = args_backmap_module[0]
    sys.argv = args_backmap_module[1]
    # Run build_db
    build_db_main()
    sys.argv = args_backmap_module[2]
    backmap_main()
    sys.argv = args_backmap_module[3]
    backmap_main()
    sys.argv = args_backmap_module[4]
    backmap_main()
    sys.argv = args_backmap_module[5]
    backmap_main()
    assert filecmp.cmp(f'{outdir}/backmapped0.pdb', f'{outdir}/backmapped1.pdb')
    assert not filecmp.cmp(f'{outdir}/backmapped0.pdb', f'{outdir}/backmapped2.pdb')
    assert not filecmp.cmp(f'{outdir}/backmapped0.pdb', f'{outdir}/backmapped3.pdb')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        u0 = mda.Universe(f'{outdir}/backmapped0.pdb')
    assert np.array_equal(u0.atoms.names, np.array(
        ['O', 'C1', 'H1', 'H2', 'C2', 'H3', 'H4', 'C3', 'H5', 'H6', 'C4', 'H7', 'H8', 'C5', 'H9', 'H10', 'C6', 'H11',
         'H12', 'C7', 'H13', 'H14', 'C8', 'H15', 'H16', 'C9', 'H17', 'H18', 'C10', 'H19', 'H20', 'C11', 'H21', 'H22',
         'H23', 'H24', 'O', 'C1',
         'H1', 'H2', 'C2', 'H3', 'H4', 'C3', 'H5', 'H6', 'C4', 'H7', 'H8', 'C5', 'H9', 'H10', 'C6', 'H11', 'H12', 'C7',
         'H13', 'H14', 'C8', 'H15', 'H16', 'C9', 'H17', 'H18', 'C10', 'H19', 'H20', 'C11', 'H21', 'H22', 'H23', 'H24',
         'H1', 'C1', 'H2', 'H3', 'C2',
         'H4', 'H5', 'C3', 'H6', 'H7', 'O', 'H8', 'H1', 'C1', 'H2', 'H3', 'C2', 'H4', 'H5', 'C3', 'H6', 'H7', 'O', 'H8',
         'O', 'C1', 'H1', 'H2', 'C2', 'H3', 'H4', 'C3', 'H5', 'H6', 'C4', 'H7', 'H8', 'C5', 'H9', 'H10', 'C6', 'H11',
         'H12', 'C7', 'H13', 'H14',
         'H15', 'H16', 'O', 'C1', 'H1', 'H2', 'C2', 'H3', 'H4', 'C3', 'H5', 'H6', 'C4', 'H7', 'H8', 'C5', 'H9', 'H10',
         'C6', 'H11', 'H12', 'C7', 'H13', 'H14', 'H15', 'H16', 'OH2', 'H1', 'H2', 'OH2', 'H1', 'H2', 'OH2', 'H1', 'H2',
         'OH2', 'H1', 'H2', 'OH2',
         'H1', 'H2', 'OH2', 'H1', 'H2', 'OH2', 'H1', 'H2', 'OH2', 'H1', 'H2']))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        u2 = mda.Universe(f'{outdir}/backmapped2.pdb')
    assert np.array_equal(u2.atoms.names, np.array(
        ['O', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'O', 'C1', 'C2', 'C3', 'C4',
         'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C1', 'C2', 'C3', 'O', 'C1', 'C2', 'C3', 'O', 'O', 'C1', 'C2',
         'C3', 'C4', 'C5', 'C6', 'C7', 'O', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'OH2', 'OH2', 'OH2', 'OH2', 'OH2',
         'OH2', 'OH2', 'OH2']))
