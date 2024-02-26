import filecmp
import os
import sys

import numpy as np
import pytest

from artsm.build_db import main
from artsm.database.db import DBdata
from fixtures import args_build_db_module, file_db0, file_db1, file_db2, file_db3


def test_build_db_module_general_functionality(args_build_db_module):
    sys.argv = args_build_db_module[0]
    db_file0 = args_build_db_module[1]
    # Run build_db
    main()
    # Check that final database is present
    assert os.path.exists(db_file0)
    path = os.path.dirname(db_file0)
    assert os.path.exists(os.path.join(path, 'release.db'))
    # Load final database and check number of frs and fr_pairs
    db = DBdata(db_file0, exist_ok=True)
    assert db.get_fr_pair_ids() == [1, 2]
    assert db.get_fr_ids() == [1, 2]
    # Check amount of data retrieved for deriving the models. Similar to artsm.model.models.derive_models
    rng = np.random.default_rng(None)
    idx = db.get_fr_pair_ids()
    for connection_id in idx:
        connection = db.get_fr_pair(connection_id, rng)
        if connection.smiles == 'CCCCCCCC':
            assert connection._X.shape[0] == 240
        elif connection.smiles == 'CCCCCCCO':
            assert connection._X.shape[0] == 480
        else:
            assert False
    idx = db.get_fr_ids()
    for fr_id in idx:
        fr = db.get_fr(fr_id, rng)
        if fr.smiles == 'CCCO':
            assert fr._coords_data.shape[0] == 240
        else:
            assert len(fr._coords_data) == 0


def test_build_db_module_seed_dp0(args_build_db_module):
    sys.argv = args_build_db_module[0]
    db_file0 = args_build_db_module[1]
    main()
    sys.argv = args_build_db_module[2]
    db_file1 = args_build_db_module[3]
    main()

    # 0 vs 1
    # db_file0 != dbfile1, since dbfile1 was restricted to 450 dp -> different model
    db0 = DBdata(db_file0, exist_ok=True)
    db1 = DBdata(db_file1, exist_ok=True)
    rng = np.random.default_rng(None)
    idx0 = db0.get_fr_pair_ids()
    idx1 = db1.get_fr_pair_ids()
    assert idx0 == idx1

    for i in idx0:
        con1 = db0.get_fr_pair(i, rng)
        con2 = db1.get_fr_pair(i, rng)
        assert con1.smiles == con2.smiles
        assert con1._model is not None
        assert con2._model is not None
        if con1.smiles == 'CCCCCCCC':
            assert con1._model.get_params() == con2._model.get_params()
            assert np.array_equal(con1._model.feature_importances_, con2._model.feature_importances_)
            assert np.array_equal(con1._main_coords1, con2._main_coords1)
            assert np.array_equal(con1._main_internal1, con2._main_internal1)
            assert np.array_equal(con1._main_coords2, con2._main_coords2)
            assert np.array_equal(con1._main_internal2, con2._main_internal2)
            assert np.array_equal(con1._main_dihedrals, con2._main_dihedrals)
        elif con1.smiles == 'CCCCCCCO':
            assert con1._model.get_params() == con2._model.get_params()
            assert not np.array_equal(con1._model.feature_importances_, con2._model.feature_importances_)
            assert not np.array_equal(con1._main_coords1, con2._main_coords1)
            assert not np.array_equal(con1._main_internal1, con2._main_internal1)
            assert not np.array_equal(con1._main_coords2, con2._main_coords2)
            assert not np.array_equal(con1._main_internal2, con2._main_internal2)
            assert not np.array_equal(con1._main_dihedrals, con2._main_dihedrals)


def test_build_db_module_seed_dp1(args_build_db_module):
    sys.argv = args_build_db_module[2]
    db_file1 = args_build_db_module[3]
    main()
    sys.argv = args_build_db_module[4]
    db_file2 = args_build_db_module[5]
    main()

    # 1 vs 2
    # db_file1 == db_file2, since both have the same seed and 450 dp
    db1 = DBdata(db_file1, exist_ok=True)
    db2 = DBdata(db_file2, exist_ok=True)
    rng = np.random.default_rng(None)
    idx1 = db1.get_fr_pair_ids()
    idx2 = db2.get_fr_pair_ids()
    assert idx1 == idx2

    for i in idx1:
        con1 = db1.get_fr_pair(i, rng)
        con2 = db2.get_fr_pair(i, rng)
        assert con1.smiles == con2.smiles
        assert con1._model is not None
        assert con2._model is not None
        assert con1._model.get_params() == con2._model.get_params()
        assert np.array_equal(con1._model.feature_importances_, con2._model.feature_importances_)
        assert np.array_equal(con1._main_coords1, con2._main_coords1)
        assert np.array_equal(con1._main_internal1, con2._main_internal1)
        assert np.array_equal(con1._main_coords2, con2._main_coords2)
        assert np.array_equal(con1._main_internal2, con2._main_internal2)
        assert np.array_equal(con1._main_dihedrals, con2._main_dihedrals)


def test_build_db_module_seed_dp2(args_build_db_module):
    sys.argv = args_build_db_module[4]
    db_file2 = args_build_db_module[5]
    main()
    sys.argv = args_build_db_module[6]
    db_file3 = args_build_db_module[7]
    main()

    # 2 vs 3
    # db_file2 != db_file3, since they have different seeds. For the fr_pair CCCCCCCC this still results in the same
    # main conformations since the randomness only plays a role if the number of dps in the database exceed the
    # specified number of dps. However, the RFRs are still different for CCCCCCCC, since they are initialized with
    # different random states.
    db2 = DBdata(db_file2, exist_ok=True)
    db3 = DBdata(db_file3, exist_ok=True)
    rng = np.random.default_rng(None)
    idx2 = db2.get_fr_pair_ids()
    idx3 = db3.get_fr_pair_ids()
    assert idx2 == idx3

    for i in idx2:
        con1 = db2.get_fr_pair(i, rng)
        con2 = db3.get_fr_pair(i, rng)
        assert con1.smiles == con2.smiles
        assert con1._model is not None
        assert con2._model is not None
        con1_params = con1._model.get_params()
        con2_params = con2._model.get_params()
        assert con1_params['random_state'] == 400
        assert con2_params['random_state'] == 500
        del con1_params['random_state']
        del con2_params['random_state']
        assert con1_params == con2_params
        if con1.smiles == 'CCCCCCCC':
            assert not np.array_equal(con1._model.feature_importances_, con2._model.feature_importances_)
            assert np.array_equal(con1._main_coords1, con2._main_coords1)
            assert np.array_equal(con1._main_internal1, con2._main_internal1)
            assert np.array_equal(con1._main_coords2, con2._main_coords2)
            assert np.array_equal(con1._main_internal2, con2._main_internal2)
            assert np.array_equal(con1._main_dihedrals, con2._main_dihedrals)
        elif con1.smiles == 'CCCCCCCO':
            assert not np.array_equal(con1._model.feature_importances_, con2._model.feature_importances_)
            assert not np.array_equal(con1._main_coords1, con2._main_coords1)
            assert not np.array_equal(con1._main_internal1, con2._main_internal1)
            assert not np.array_equal(con1._main_coords2, con2._main_coords2)
            assert not np.array_equal(con1._main_internal2, con2._main_internal2)
            assert not np.array_equal(con1._main_dihedrals, con2._main_dihedrals)


def test_get_connection(args_build_db_module):
    sys.argv = args_build_db_module[0]
    db_file = args_build_db_module[1]
    main()

    # Check function get_connection 450 dp
    db = DBdata(db_file, exist_ok=True, n_datapoints=450)
    rng = np.random.default_rng(None)
    idx = db.get_fr_pair_ids()
    for connection_id in idx:
        connection = db.get_fr_pair(connection_id, rng)
        if connection.smiles == 'CCCCCCCC':
            assert connection._X.shape[0] == 240
        elif connection.smiles == 'CCCCCCCO':
            assert connection._X.shape[0] == 450
        else:
            pytest.fail('SMILES are not correct.')

    # Check function get_connection 50 dp
    db = DBdata(db_file, exist_ok=True, n_datapoints=50)
    rng = np.random.default_rng(None)
    idx = db.get_fr_pair_ids()
    for connection_id in idx:
        connection = db.get_fr_pair(connection_id, rng)
        assert connection._X.shape[0] == 50

