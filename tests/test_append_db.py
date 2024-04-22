import filecmp
import os.path
import sys

import numpy as np

from artsm.build_db import main as build_db_main
from artsm.append_db import main as append_db_main
from artsm.database.db import DBdata
from fixtures import args_append_db_module, file_db0, file_db1


def compare_none_or_array(arr1, arr2):
    if arr1 is None and arr2 is None:
        return True
    elif arr1 is not None and arr2 is None:
        return False
    elif arr1 is None and arr2 is not None:
        return False
    else:
        return np.array_equal(arr1, arr2)


def compare_frs(fr1, fr2):
    """
    Compare attributes of Fragment fr1 and fr2.
    """
    # Compare general characteristics
    assert fr1.smiles == fr2.smiles
    assert np.array_equal(fr1.atoms, fr2.atoms)
    assert fr1.A.equals(fr2.A)
    assert np.array_equal(fr1.zmatrix, fr2.zmatrix)

    # Compare derived models
    compare_none_or_array(fr1._main_coords, fr2._main_coords)
    if fr1._model is None:
        assert fr2._model is None
    elif fr2._model is None:
        assert fr1._model is None
    else:
        assert np.array_equal(fr1._model.labels, fr2._model.labels)
        assert np.array_equal(fr1._model.p, fr2._model.p)

    # Compare data
    compare_none_or_array(fr1._zmatrix_data, fr2._zmatrix_data)
    compare_none_or_array(fr1._coords_data, fr2._coords_data)


def compare_frs_dbs(db1, db2, identifier, rng):
    """
    Get both fragments with a specified identifier from db1 and db2 and compare their attributes.
    """
    fr_db1 = db1.get_fr(identifier, rng)
    fr_db2 = db2.get_fr(identifier, rng)
    compare_frs(fr_db1, fr_db2)


def compare_fr_pairs(fr_pair1, fr_pair2):
    """
    Compare attributes of FragmentPair fr_pair1 and fr_pair2.
    """
    # Compare general characteristics
    compare_frs(fr_pair1.fr1, fr_pair2.fr1)
    compare_frs(fr_pair1.fr2, fr_pair2.fr2)
    assert fr_pair1.smiles == fr_pair2.smiles

    # Compare derived models
    if fr_pair1._model is None:
        assert fr_pair2._model is None
    elif fr_pair2._model is None:
        assert fr_pair1._model is None
    else:
        assert fr_pair1._model.get_params() == fr_pair2._model.get_params()
        assert np.array_equal(fr_pair1._model.feature_importances_, fr_pair2._model.feature_importances_)
    compare_none_or_array(fr_pair1._main_coords1, fr_pair2._main_coords1)
    compare_none_or_array(fr_pair1._main_internal1, fr_pair2._main_internal1)
    compare_none_or_array(fr_pair1._main_coords2, fr_pair2._main_coords2)
    compare_none_or_array(fr_pair1._main_internal2, fr_pair2._main_internal2)
    compare_none_or_array(fr_pair1._main_dihedrals, fr_pair2._main_dihedrals)

    # Compare data
    np.array_equal(fr_pair1._fr1_zmatrix_data, fr_pair2._fr1_zmatrix_data)
    np.array_equal(fr_pair1._fr1_coords_data, fr_pair2._fr1_coords_data)
    np.array_equal(fr_pair1._fr2_zmatrix_data, fr_pair2._fr2_zmatrix_data)
    np.array_equal(fr_pair1._fr2_coords_data, fr_pair2._fr2_coords_data)
    np.array_equal(fr_pair1._X, fr_pair2._X)
    np.array_equal(fr_pair1._dihedral_data, fr_pair2._dihedral_data)


def compare_fr_pairs_dbs(db1, db2, identifier, rng):
    """
    Get both fragment pairs with a specified identifier from db1 and db2 and compare their attributes.
    """
    fr_pair1 = db1.get_fr_pair(identifier, rng)
    fr_pair2 = db2.get_fr_pair(identifier, rng)
    compare_fr_pairs(fr_pair1, fr_pair2)
    
    
def test_append_db_module_general_functionality(args_append_db_module):
    """
    What is tested?
    * the existence of database files after generation by the
      build_db and append_db scripts
    * the presence of newly added fragment pair in the appended database
        * check counts of fragment pair's _X attribute
    * lists of existing and newly added fragment pairs are determined
        * validity of the existing attributes are checked using compare_frs_dbs()
        * we check if important attributes exist for newly added fragment pairs
    * lists of existing and newly added fragments are determined
        * validity of the existing attributes are checked using compare_frs_dbs()
        * we check if important attributes exist for newly added fragments
    """
    # Build database with build_db and load it
    rng = np.random.default_rng(None)
    sys.argv = args_append_db_module[0]
    db_file0 = args_append_db_module[1]
    build_db_main()
    assert os.path.exists(db_file0)

    old_db = DBdata(db_file0, exist_ok=True)
    fr_pairs_in_old_db = old_db.get_fr_pair_ids()  # [1]
    fragments_in_old_db = old_db.get_fr_ids()

    # Append database and load it
    sys.argv = args_append_db_module[2]
    out_file1 = args_append_db_module[3]
    append_db_main()
    assert os.path.exists(out_file1)

    appended_db = DBdata(out_file1, exist_ok=True)
    fr_pairs_in_new_db = appended_db.get_fr_pair_ids()  # [1, 2]
    fragments_in_new_db = appended_db.get_fr_ids()

    # Compare old to appended database.
    assert fr_pairs_in_old_db == [1]
    assert fragments_in_old_db == [1, 2]
    assert fr_pairs_in_new_db == [1, 2]
    assert fragments_in_new_db == [1, 2]

    compare_fr_pairs_dbs(old_db, appended_db, 1, rng)
    compare_frs_dbs(old_db, appended_db, 1, rng)
    compare_frs_dbs(old_db, appended_db, 2, rng)

    new_fr_pair = appended_db.get_fr_pair(2, rng)
    assert new_fr_pair.smiles == 'CCCCCCCC'
    assert new_fr_pair._X.shape[0] == 240
    assert new_fr_pair._model is not None
    assert new_fr_pair.fr1.smiles == 'CCCC'
    assert new_fr_pair.fr2.smiles == 'CCCC'


def test_append_db_module_check_no_append_case(args_append_db_module):
    """
    In this test, we check if no changes are made to the appended db when we use the
    same simulation file as input which was used to generate the source database.
    """
    # Build database with build_db and load it
    rng = np.random.default_rng(None)
    sys.argv = args_append_db_module[6]
    db_file0 = args_append_db_module[1]
    build_db_main()
    assert os.path.exists(db_file0)
    old_db = DBdata(db_file0, exist_ok=True)

    # the datapoint counts for bonds and angles are set to 1000.
    # No new data points should be added in the append step.
    old_db._cursor.execute('''UPDATE angles SET datapoints = 1000 
                                         WHERE element1 = 'C' AND element2 = 'C' AND element3 = 'O' 
                                         AND bond_type1 = 1 AND bond_type2 = 1''')
    old_db._cursor.execute('''UPDATE bonds SET datapoints = 1000
                                         WHERE element1 = 'C' AND element2 = 'O' AND bond_type = 1''')
    old_db._connection.commit()

    # Try to append the database with the same config.yaml file. No changes are expected.
    sys.argv = args_append_db_module[2]
    appended_db_path = args_append_db_module[3]
    append_db_main()
    assert os.path.exists(appended_db_path)
    appended_db = DBdata(appended_db_path, exist_ok=True)

    # Assertions
    # Compare attributes of all fr_pairs before and after append operation
    for identifier in old_db.get_fr_pair_ids():
        compare_fr_pairs_dbs(old_db, appended_db, identifier, rng)
    for identifier in appended_db.get_fr_pair_ids():
        compare_fr_pairs_dbs(old_db, appended_db, identifier, rng)
    # Compare attributes of all fr_pairs before and after append operation
    for identifier in old_db.get_fr_ids():
        compare_frs_dbs(old_db, appended_db, identifier, rng)
    for identifier in appended_db.get_fr_ids():
        compare_frs_dbs(old_db, appended_db, identifier, rng)
    # Check angle and bond counts
    angle_counts_after_append = appended_db.get_angle_counts().values()
    bond_counts_after_append = appended_db.get_bond_counts().values()
    assert list(angle_counts_after_append) == [1000, 1000]
    assert list(bond_counts_after_append) == [1000, 1000]


def test_append_db_module_angle_and_bond_counts(args_append_db_module):
    """
    In this test, we mainly check if the angle and bond counts are updated correctly.
    We check for two cases:
    * when the count is below 1000
    * when the count is above or equal to 1000

    The append_db script is run twice on the same database file.
    """
    # Build database with build_db and load it
    sys.argv = args_append_db_module[0]
    db_file0 = args_append_db_module[1]
    build_db_main()
    assert os.path.exists(db_file0)
    old_db = DBdata(db_file0, exist_ok=True)

    # Append database and load it
    sys.argv = args_append_db_module[2]
    out_file1 = args_append_db_module[3]
    append_db_main()
    assert os.path.exists(out_file1)
    appended_db = DBdata(out_file1, exist_ok=True)

    # Load and verify old bond and angle values
    old_angle_counts = old_db.get_angle_counts().values()
    old_bond_counts = old_db.get_bond_counts().values()
    assert list(old_angle_counts) == [950, 190]
    assert list(old_bond_counts) == [1000, 190]

    # Load and verify new bond and angle values
    new_angle_counts = appended_db.get_angle_counts().values()
    new_bond_counts = appended_db.get_bond_counts().values()
    assert list(new_angle_counts) == [1950, 910]
    assert list(new_bond_counts) == [1000, 910]

    # Set all bonds and angle data points to at least 1000
    old_db._cursor.execute('''UPDATE angles SET datapoints = 1000 
                                             WHERE element1 = 'C' AND element2 = 'C' AND element3 = 'O' 
                                             AND bond_type1 = 1 AND bond_type2 = 1''')
    old_db._cursor.execute('''UPDATE bonds SET datapoints = 1000
                                             WHERE element1 = 'C' AND element2 = 'O' AND bond_type = 1''')
    old_db._connection.commit()

    # run append_db again and load the database
    append_db_main()
    assert os.path.exists(out_file1)
    appended_db = DBdata(out_file1, exist_ok=True)
    new_angle_counts = appended_db.get_angle_counts().values()
    new_bond_counts = appended_db.get_bond_counts().values()
    assert list(new_angle_counts) == [1950, 1000]
    assert list(new_bond_counts) == [1000, 1000]


def test_append_db_module_release_file(args_append_db_module):
    """
    In this test, we check if the release.db file is generated correctly.
    We check if the release file has the correct attributes and if the data tables are empty.
    """
    # Build database with build_db and load it
    rng = np.random.default_rng(None)
    sys.argv = args_append_db_module[0]
    db_file0 = args_append_db_module[1]
    build_db_main()
    assert os.path.exists(db_file0)
    old_db = DBdata(db_file0, exist_ok=True)

    # Append database and load it
    sys.argv = args_append_db_module[4]
    appended_file_path = args_append_db_module[5]
    append_db_main()
    assert os.path.exists(appended_file_path)
    appended_db = DBdata(appended_file_path, exist_ok=True)

    # check that release db file exists and load it
    release_file_path = os.path.join(os.path.dirname(appended_file_path), 'release.db')
    assert os.path.exists(release_file_path)
    release_db = DBdata(release_file_path, exist_ok=True)

    # Assert fr_pair and fr ids
    fr_pairs_in_old_db = old_db.get_fr_pair_ids()
    fr_pairs_in_appended_db = appended_db.get_fr_pair_ids()
    fr_pairs_in_release_db = release_db.get_fr_pair_ids()
    assert fr_pairs_in_old_db == [1]
    assert fr_pairs_in_appended_db == [1, 2]
    assert fr_pairs_in_release_db == [1, 2]
    frs_in_old_db = old_db.get_fr_ids()
    frs_in_appended_db = appended_db.get_fr_ids()
    frs_in_release_db = release_db.get_fr_ids()
    assert frs_in_old_db == [1, 2]
    assert frs_in_appended_db == [1, 2]
    assert frs_in_release_db == [1, 2]

    # Check that no data is available in the release data base. Model and general information should still be present.
    # Fragment pairs
    fr_pair1 = release_db.get_fr_pair(1, rng)
    assert fr_pair1.fr1.smiles == 'CCCO'
    assert fr_pair1.fr2.smiles == 'CCCC'
    assert fr_pair1.smiles == 'CCCCCCCO'
    for mod in fr_pair1.get_models():
        assert mod is not None
    for data in fr_pair1.get_data():
        assert list(data) == []

    fr_pair2 = release_db.get_fr_pair(2, rng)
    assert fr_pair2.fr1.smiles == 'CCCC'
    assert fr_pair2.fr2.smiles == 'CCCC'
    assert fr_pair2.smiles == 'CCCCCCCC'
    for mod in fr_pair2.get_models():
        assert mod is not None
    for data in fr_pair2.get_data():
        assert list(data) == []

    fr1 = release_db.get_fr(1, rng)
    assert fr1.smiles == 'CCCO'
    for mod in fr1.get_models():
        assert mod is None
    for data in fr1.get_data():
        assert list(data) == []

    fr2 = release_db.get_fr(2, rng)
    assert fr2.smiles == 'CCCC'
    for mod in fr2.get_models():
        assert mod is None
    for data in fr2.get_data():
        assert list(data) == []

