import re
import sys
import warnings

import MDAnalysis as mda
import numpy as np
import pytest

from artsm.coarse_grain import main
from fixtures import args_cg_module, file_cg0, file_cg1


def test_cg0(args_cg_module):
    sys.argv = args_cg_module[0]
    main()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        u = mda.Universe(args_cg_module[1])
    assert np.array_equal(u.atoms.names, np.array(['C1O', 'C2A', 'C3A', 'C1O', 'C2A', 'C3A', 'C1A', 'C1A',
                                                   'C1O', 'C2A', 'C1O', 'C2A']))
    assert np.array_equal(u.residues.resnames, np.array(['UND', 'UND', 'PRP', 'PRP', 'HEP', 'HEP']))
    coords = np.array(
        [[20.91, 23.821, 13.001],
         [23.532, 24.296, 16.957],
         [22.702, 25.277, 21.248],
         [17.383, 23.028, 31.511],
         [13.311, 24.596, 32.832],
         [10.458, 23.675, 29.895],
         [0.513, 25.419, 23.874],
         [30.123, 11.375, 7.437],
         [34.634, 35.922, 11.752],
         [33.961, 33.895, 7.689],
         [32.319, 5.072, 25.9],
         [36.813, 4.365, 27.294]], dtype=np.float32)
    assert np.array_equal(u.atoms.positions, coords)


def test_cg1(args_cg_module, caplog):
    sys.argv = args_cg_module[2]
    main()
    pattern = 'It seems you have requested an output trajectory without providing an input trajectory. ' \
              'I can thus only provide an output snapshot.'
    assert re.search(pattern, caplog.text)


def test_cg2(args_cg_module):
    sys.argv = args_cg_module[4]
    main()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        u = mda.Universe(args_cg_module[1], args_cg_module[3])
    assert u.trajectory.n_frames == 6000


def test_cg3(args_cg_module, caplog):
    sys.argv = args_cg_module[5]
    with pytest.raises(SystemExit):
        main()
    pattern = 'Writing to output file failed. Maybe the following went wrong:'
    assert re.search(pattern, caplog.text)
