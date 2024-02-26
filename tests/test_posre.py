import os
import sys
import warnings

import MDAnalysis as mda
import numpy as np

from artsm.generate_posre import main
from fixtures import args_posre_module, file_posre


def equal_lines(arr):
    assert np.all(arr[:, 0] == arr[0, 0])
    assert np.all(arr[:, 1] == arr[0, 1])
    assert np.all(arr[:, 2] == arr[0, 2])


def test_posre0(args_posre_module):
    sys.argv = args_posre_module[0]
    main()

    # Check itp
    path = os.path.dirname(args_posre_module[1])
    itp = os.path.basename(args_posre_module[1])

    und_itp = os.path.join(path, f'UND_{itp}')
    with open(und_itp) as file_:
        assert file_.readline().strip() == '[ position_restraints ]'
        assert file_.readline().strip() == ';ai    func    g    r    k'
        ids, funcs, g, r, k = zip(*[line.strip().split() for line in file_])
    assert ids == tuple(['1', '2', '5', '8', '11', '14', '17', '20', '23', '26', '29', '32'])
    assert funcs == tuple(['2' for _ in range(12)])
    assert g == tuple(['1' for _ in range(12)])
    assert r == tuple(['0.38' for _ in range(12)])
    assert k == tuple(['1000' for _ in range(12)])

    hep_itp = os.path.join(path, f'HEP_{itp}')
    with open(hep_itp) as file_:
        assert file_.readline().strip() == '[ position_restraints ]'
        assert file_.readline().strip() == ';ai    func    g    r    k'
        ids, funcs, g, r, k = zip(*[line.strip().split() for line in file_])
    assert ids == tuple(['1', '2', '5', '8', '11', '14', '17', '20'])
    assert funcs == tuple(['2' for _ in range(8)])
    assert g == tuple(['1' for _ in range(8)])
    assert r == tuple(['0.38' for _ in range(8)])
    assert k == tuple(['1000' for _ in range(8)])

    prp_itp = os.path.join(path, f'PRP_{itp}')
    with open(prp_itp) as file_:
        assert file_.readline().strip() == '[ position_restraints ]'
        assert file_.readline().strip() == ';ai    func    g    r    k'
        ids, funcs, g, r, k = zip(*[line.strip().split() for line in file_])
    assert ids == tuple(['2', '5', '8', '11'])
    assert funcs == tuple(['2' for _ in range(4)])
    assert g == tuple(['1' for _ in range(4)])
    assert r == tuple(['0.38' for _ in range(4)])
    assert k == tuple(['1000' for _ in range(4)])

    tip_itp = os.path.join(path, f'TIP_{itp}')
    with open(tip_itp) as file_:
        assert file_.readline().strip() == '[ position_restraints ]'
        assert file_.readline().strip() == ';ai    func    g    r    k'
        ids, funcs, g, r, k = zip(*[line.strip().split() for line in file_])
    assert ids == tuple(['1'])
    assert funcs == ('2', )
    assert g == ('1', )
    assert r == ('0.38', )
    assert k == ('1000', )

    # Check pdb for some beads
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        u = mda.Universe(args_posre_module[2])
    assert len(u.atoms) == 168
    assert np.array_equal(u.residues.resnames, ['UND', 'UND', 'PRP', 'PRP', 'HEP', 'HEP',
                                                'TIP', 'TIP', 'TIP', 'TIP', 'TIP', 'TIP', 'TIP', 'TIP'])
    coords_und = u.select_atoms(
        'resid 1 and (name O or name H24 or name C1 or name H1 or name H2 or name C2 or name H3 '
        'or name H4 or name C3 or name H5 or name H6)').positions
    equal_lines(coords_und)
    coords_prp = u.select_atoms('resid 3').positions
    equal_lines(coords_prp)
    coords_tip = u.select_atoms('resid 7').positions
    equal_lines(coords_tip)
