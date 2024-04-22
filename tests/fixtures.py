import os
import pickle
import pytest
import sqlite3

import numpy as np
import pandas as pd


#  CLI
@pytest.fixture
def args_build_db():
    args0 = '-d database.db -g file1.yaml --time_step 400 --seed 300 --n_datapoints 200'.split()  # Valid
    args1 = '-d database.db -c file1.yaml --time_step 400 --seed 300 --n_datapoints 200'.split()  # Valid
    args2 = '-d database.db -s file1.pdb -x file2.xtc -t file3.yaml --time_step 400 --seed 300 --n_datapoints 200'.split()  # Valid
    args3 = '-d database.db -g file1.yaml'.split()  # Valid. Check default values
    args4 = '-d database.db -s file1.pdb -x file2.xtc -t file3.yaml -t file4.yaml -t file5.yaml'.split()  # Valid. Check multiple -t. This results in a warning later.
    args5 = '-d database.db -s file1.pdb -x file2.xtc -t file3.yaml file4.yaml file5.yaml'.split()  # Valid. Check multiple -t
    args6 = '-d database.db -g file1.yaml --time_step 400 --seed 300 --n_datapoints 200 -s file2.pdb -t file3.yaml'.split()  # Ignore -s and -t
    args7 = '-d database.db -s file1.pdb -t file3.yaml --time_step 400 --seed 300 --n_datapoints 200'.split()  # Error: -x missing
    args8 = '-g file1.yaml'  # Error no database
    return [args0, args1, args2, args3, args4, args5, args6, args7, args8]


@pytest.fixture
def args_append_db():
    args0 = '-d database.db -g file1.yaml --time_step 400 --seed 300 -o appended1.db'.split()
    args1 = '-d database.db -g file1.yaml --time_step 400 --seed 300 -o appended2.db --release'.split()
    return [args0, args1]


@pytest.fixture
def args_backmap():
    args0 = '-d database.db -s file1.pdb -t file2.yaml -o file3.pdb --seed 300 --hydrogens'.split()  # Valid
    args1 = '-d database.db -s file1.pdb -t file2.yaml -o file3.pdb'.split()  # Valid
    args2 = '-d database.db -s file1.pdb -t file2.yaml file3.yaml file4.yaml -o file5.pdb'.split()  # Valid. Check multiple -t
    args3 = '-d database.db -s file1.pdb -t file2.yaml -t file3.yaml -t file4.yaml -o file5.pdb'.split()  # Valid. Check multiple -t
    args4 = '-d database.db -t file2.yaml -t file3.yaml -t file4.yaml -o file5.pdb'.split()  # Fail. -s missing
    args5 = ['']  # Fail. Everything is missing
    return [args0, args1, args2, args3, args4, args5]


@pytest.fixture
def args_mapping():
    args0 = '-a file1.pdb -m file2.pdb -s UND CCCC -o file3.yaml'.split()  # Valid
    args1 = '-a file1.pdb -m file2.pdb -s UND CCCC -s PRO CCCO -o file3.yaml'.split()  # Valid. Multiple -s
    args2 = '-a file1.pdb -s UND CCCC -o file3.yaml'.split()  # Error. -m missing
    args3 = '-a file1.pdb -m file2.pdb -s CCCO -o file3.yaml'.split()  # Error. Wrong no identifier for SMILE
    return [args0, args1, args2, args3]


@pytest.fixture()
def args_cg():
    args0 = '-a file1.pdb -x file2.xtc -t file3.yaml -o file4.pdb file5.xtc'.split()  # Valid
    args1 = '-a file1.pdb -t file2.yaml -o file3.pdb'.split()  # Valid
    args2 = '-a file1.pdb -x file2.xtc -t file3.yaml file4.yaml file5.yaml -o file6.pdb'.split()  # Valid. Multiple -t
    args3 = '-x file1.xtc -t file2.yaml file3.yaml file4.yaml -o file5.pdb file6.pdb'.split()  # Error. -a missing
    return [args0, args1, args2, args3]


@pytest.fixture
def args_posre():
    args0 = '-c file1.pdb -t file2.yaml -i file3.itp -r file4.pdb'.split()  # Valid
    args1 = '-c file1.pdb -t file2.yaml file3.yaml file4.yaml -i file5.itp -r file6.pdb'.split()  # Valid. Multiple -t
    args2 = '-c file1.pdb -t file2.yaml'.split()  # Valid. Minimal
    args3 = '-c file1.pdb'.split()  # Error. -t missing
    return [args0, args1, args2, args3]


@pytest.fixture
def file_db0(tmp_path):
    db_dir = tmp_path / 'db1'
    db_dir.mkdir(exist_ok=True)
    filename = f'{tmp_path.resolve()}/db1/database.db'
    return filename


@pytest.fixture
def file_db1(tmp_path):
    db_dir = tmp_path / 'db2'
    db_dir.mkdir(exist_ok=True)
    filename = f'{tmp_path.resolve()}/db2/database.db'
    return filename


@pytest.fixture
def file_db2(tmp_path):
    db_dir = tmp_path / 'db3'
    db_dir.mkdir(exist_ok=True)
    filename = f'{tmp_path.resolve()}/db3/database.db'
    return filename


@pytest.fixture
def file_db3(tmp_path):
    db_dir = tmp_path / 'db4'
    db_dir.mkdir(exist_ok=True)
    filename = f'{tmp_path.resolve()}/db4/database.db'
    return filename


@pytest.fixture
def file_mapping0(tmp_path):
    db_dir = tmp_path / 'mapping0'
    db_dir.mkdir(exist_ok=True)
    filename = f'{tmp_path.resolve()}/mapping0/mapping.yaml'
    return filename


@pytest.fixture
def file_cg0(tmp_path):
    db_dir = tmp_path / 'cg0'
    db_dir.mkdir(exist_ok=True)
    filename = f'{tmp_path.resolve()}/cg0/cg.pdb'
    return filename


@pytest.fixture
def file_cg1(tmp_path):
    db_dir = tmp_path / 'cg0'
    db_dir.mkdir(exist_ok=True)
    filename = f'{tmp_path.resolve()}/cg0/cg.xtc'
    return filename


@pytest.fixture
def file_posre(tmp_path):
    db_dir = tmp_path / 'posre'
    db_dir.mkdir(exist_ok=True)
    filename_itp = f'{tmp_path.resolve()}/posre/posre.itp'
    filename_pdb = f'{tmp_path.resolve()}/posre/posre.pdb'
    return filename_itp, filename_pdb


# parse_args
@pytest.fixture
def args_build_db_data():
    args0 = f'-d database.db -s data/atomistic_cg_data/atomistic_mixture.pdb -x data/atomistic_cg_data/atomistic_mixture.xtc ' \
            f'-t data/mapping_files/und.yaml data/mapping_files/hep.yaml data/mapping_files/prp.yaml ' \
            f'data/mapping_files/water.yaml --time_step 400 --seed 300 --n_datapoints 200'.split()  # Valid s, t, x
    args1 = f'-d database.db -c data/config/config1.yaml'.split()  # Valid c
    args2 = f'-d database.db -c data/config/config2.yaml'.split()  # Valid c. Different t
    args3 = f'-d database.db -g data/config/config_global1.yaml'.split()  # Valid g.
    args4 = f'-d database.db -g data/config/config_global2.yaml --time_step 40'.split()  # Valid g. Multiple sim. Different time_steps.
    args5 = f'-d database.db -c data/config/config5.yaml'.split()  # Valid, but random key 'random'

    return [args0, args1, args2, args3, args4, args5]


@pytest.fixture
def args_append_db_module(file_db0):
    path0 = os.path.dirname(file_db0)
    # args for build_db.py file
    args0 = f'file.py -d {file_db0} -c data/config/config_heptanol.yaml --seed 400'.split()

    # NOTE: file_db0 is the path to the database generated by build_db()
    # and args0 are the arguments to the build_db() script

    # modify path of file_db0 to form out_db1 and out_db2 (generated by append_db.py)
    out_db1 = os.path.join(path0, 'appended1.db')
    out_db2 = os.path.join(path0, 'appended2.db')

    # args/command for append_db.py
    args1 = f'file.py -d {file_db0} -g data/config/config_global1.yaml --seed 400 -o {out_db1}'.split()
    args2 = f'file.py -d {file_db0} -g data/config/config_global1.yaml --seed 400 -o {out_db2} --release'.split()

    # args for build_db
    args3 = f'file.py -d {file_db0} -g data/config/config_global1.yaml --seed 400'.split()
    return [args0, f'{file_db0}',
            args1, f'{out_db1}',
            args2, f'{out_db2}',
            args3]


@pytest.fixture
def args_backmap_data():
    args0 = '-d database.db -s data/atomistic_cg_data/cg_mixture.pdb -t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            'data/mapping_files/prp.yaml data/mapping_files/water.yaml -o backmapped.pdb ' \
            '--seed 300 --hydrogens'.split()  # Valid
    return [args0]


@pytest.fixture
def args_build_db_module(file_db0, file_db1, file_db2, file_db3):
    args0 = f'file.py -d {file_db0} -g data/config/config_global1.yaml --seed 400'.split()
    args1 = f'file.py -d {file_db1} -g data/config/config_global1.yaml --seed 400 --n_datapoints 450'.split()
    args2 = f'file.py -d {file_db2} -g data/config/config_global1.yaml --seed 400 --n_datapoints 450'.split()
    args3 = f'file.py -d {file_db3} -g data/config/config_global1.yaml --seed 500 --n_datapoints 450'.split()

    return [args0, f'{file_db0}', args1, f'{file_db1}', args2, f'{file_db2}', args3, f'{file_db3}']


@pytest.fixture
def args_backmap_module(file_db0, file_db1):
    path0 = os.path.dirname(file_db0)
    args0 = f'file.py -d {file_db0} -g data/config/config_global1.yaml --seed 400'.split()
    args1 = f'file.py -d {file_db0} -s data/atomistic_cg_data/cg_mixture.pdb -t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml -o {path0}/backmapped0.pdb --seed 20 --hydrogens'.split()
    args2 = f'file.py -d {file_db0} -s data/atomistic_cg_data/cg_mixture.pdb -t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml -o {path0}/backmapped1.pdb --seed 20 --hydrogens'.split()
    args3 = f'file.py -d {file_db0} -s data/atomistic_cg_data/cg_mixture.pdb -t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml -o {path0}/backmapped2.pdb --seed 20'.split()
    args4 = f'file.py -d {file_db0} -s data/atomistic_cg_data/cg_mixture.pdb -t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml -o {path0}/backmapped3.pdb --seed 40 --hydrogens'.split()
    return [path0, args0, args1, args2, args3, args4]


@pytest.fixture
def args_mapping_module(file_mapping0):
    args0 = f'file.py -a data/atomistic_cg_data/atomistic_mixture.pdb -m data/atomistic_cg_data/cg_mixture.pdb -s UND CCCCCCCCCCCO ' \
            f'-s HEP CCCCCCCO -s PRP CCCO -s TIP O -o {file_mapping0}'.split()  # General functionality
    args1 = f'file.py -a data/atomistic_cg_data/atomistic_mixture.pdb -m data/atomistic_cg_data/cg_mixture.pdb -s UND CCCCCCCCCCCO ' \
            f'-s HEP CCCCCCCO -s PRP CCCO -s TIP O -w TIP4P -o {file_mapping0}'.split()  # Check -w
    return [args0, f'{file_mapping0}', args1]


@pytest.fixture
def args_cg_module(file_cg0, file_cg1):
    args0 = f'file.py -a data/atomistic_cg_data/atomistic_mixture.pdb -t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml -o {file_cg0}'.split()  # Valid
    # Valid. Warning for providing xtc as output but not input
    args1 = f'file.py -a data/atomistic_cg_data/atomistic_mixture.pdb -t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml -o {file_cg0} {file_cg1}'.split()
    args2 = f'file.py -a data/atomistic_cg_data/atomistic_mixture.pdb -x data/atomistic_cg_data/atomistic_mixture.xtc ' \
            f'-t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml -o {file_cg0} {file_cg1}'.split()  # Valid
    args3 = f'file.py -a data/atomistic_cg_data/atomistic_mixture.pdb -x data/atomistic_cg_data/atomistic_mixture.xtc ' \
            f'-t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml ' \
            f'-o {file_cg0} {file_cg1} arb1.txt arb2.txt'.split()  # Error. Too many arguments for -o
    return [args0, f'{file_cg0}', args1, f'{file_cg1}', args2, args3]


@pytest.fixture
def args_posre_module(file_posre):
    args0 = f'file.py -c data/atomistic_cg_data/cg_mixture.pdb -t data/mapping_files/und.yaml data/mapping_files/hep.yaml ' \
            f'data/mapping_files/prp.yaml data/mapping_files/water.yaml -i {file_posre[0]} -r {file_posre[1]}'.split()
    return [args0, f'{file_posre[0]}', f'{file_posre[1]}']


@pytest.fixture
def args_clashing():
    box_dims = np.array([5, 5, 5, 90, 90, 90])
    coords1 = np.array([[0, 0, 0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0]])
    coords2 = np.array([[1.543543, 0., 0.], [1.543543, 0., 0.]])
    coords3 = np.array([[0, 0, 0], [0.16, 0.04, 0.011314], [0.24, 0.06, 0.016971], [0.4, 0.1, 0.028285]])
    return [box_dims, coords1, coords2, coords3]
