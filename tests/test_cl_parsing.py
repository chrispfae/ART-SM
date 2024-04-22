import re

import pytest

import artsm.utils.cli as cli
from fixtures import args_build_db, args_backmap, args_mapping, args_cg, args_posre, args_append_db


def test_build_db_cli_0(args_build_db):
    res = vars(cli.parse_cl_db(args_build_db[0]))
    assert res['d'] == 'database.db'
    assert res['g'] == 'file1.yaml'
    assert res['c'] is None
    assert res['s'] is None
    assert res['t'] is None
    assert res['x'] is None
    assert res['time_step'] == 400
    assert res['seed'] == 300
    assert res['n_datapoints'] == 200


def test_build_db_cli_1(args_build_db):
    res = vars(cli.parse_cl_db(args_build_db[1]))
    assert res['d'] == 'database.db'
    assert res['g'] is None
    assert res['c'] == 'file1.yaml'
    assert res['s'] is None
    assert res['t'] is None
    assert res['x'] is None
    assert res['time_step'] == 400
    assert res['seed'] == 300
    assert res['n_datapoints'] == 200


def test_build_db_cli_2(args_build_db):
    res = vars(cli.parse_cl_db(args_build_db[2]))
    assert res['d'] == 'database.db'
    assert res['g'] is None
    assert res['c'] is None
    assert res['s'] == 'file1.pdb'
    assert res['t'] == ['file3.yaml']
    assert res['x'] == 'file2.xtc'
    assert res['time_step'] == 400
    assert res['seed'] == 300
    assert res['n_datapoints'] == 200


def test_build_db_cli_3(args_build_db):
    res = vars(cli.parse_cl_db(args_build_db[3]))
    assert res['d'] == 'database.db'
    assert res['g'] == 'file1.yaml'
    assert res['c'] is None
    assert res['s'] is None
    assert res['t'] is None
    assert res['x'] is None
    assert res['time_step'] == 500
    assert res['seed'] is None
    assert res['n_datapoints'] == 500


def test_build_db_cli_4(args_build_db):
    res = vars(cli.parse_cl_db(args_build_db[4]))
    assert res['d'] == 'database.db'
    assert res['g'] is None
    assert res['c'] is None
    assert res['s'] == 'file1.pdb'
    assert res['t'] == ['file3.yaml', 'file4.yaml', 'file5.yaml']
    assert res['x'] == 'file2.xtc'
    assert res['time_step'] == 500
    assert res['seed'] is None
    assert res['n_datapoints'] == 500


def test_build_db_cli_5(args_build_db):
    res = vars(cli.parse_cl_db(args_build_db[5]))
    assert res['d'] == 'database.db'
    assert res['g'] is None
    assert res['c'] is None
    assert res['s'] == 'file1.pdb'
    assert res['t'] == ['file3.yaml', 'file4.yaml', 'file5.yaml']
    assert res['x'] == 'file2.xtc'
    assert res['time_step'] == 500
    assert res['seed'] is None
    assert res['n_datapoints'] == 500


def test_build_db_cli_6(args_build_db, caplog):
    res = vars(cli.parse_cl_db(args_build_db[6]))
    assert res['d'] == 'database.db'
    assert res['g'] == 'file1.yaml'
    assert res['c'] is None
    assert res['s'] is None
    assert res['t'] is None
    assert res['x'] is None
    assert res['time_step'] == 400
    assert res['seed'] == 300
    assert res['n_datapoints'] == 200

    # Catch warning
    assert caplog.text == 'WARNING  artsm.utils.cli:cli.py:67 Option -g is provided. Additional argument -s is ignored. Please provide it in the individual simulation config files.\n' \
                          'WARNING  artsm.utils.cli:cli.py:67 Option -g is provided. Additional argument -t is ignored. Please provide it in the individual simulation config files.\n'


def test_build_db_cli_7(args_build_db, caplog):
    with pytest.raises(SystemExit):
        cli.parse_cl_db(args_build_db[7])
    pattern = 'Option -x is missing'
    assert re.search(pattern, caplog.text)


def test_build_db_cli_8(args_build_db):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cl_db(args_build_db[8])
    assert exc.value.code == 2


def test_append_db_cli_0(args_append_db):
    res = vars(cli.parse_cl_append(args_append_db[0]))
    assert res['d'] == 'database.db'
    assert res['g'] == 'file1.yaml'
    assert res['c'] is None
    assert res['s'] is None
    assert res['t'] is None
    assert res['x'] is None
    assert res['time_step'] == 400
    assert res['seed'] == 300
    assert res['o'] == 'appended1.db'


def test_append_db_cli_1(args_append_db):
    res = vars(cli.parse_cl_append(args_append_db[1]))
    assert res['d'] == 'database.db'
    assert res['g'] == 'file1.yaml'
    assert res['c'] is None
    assert res['s'] is None
    assert res['t'] is None
    assert res['x'] is None
    assert res['time_step'] == 400
    assert res['seed'] == 300
    assert res['o'] == 'appended2.db'
    assert res['release']


def test_backmap_cli_0(args_backmap):
    res = vars(cli.parse_cl_backmap(args_backmap[0]))
    assert res['d'] == 'database.db'
    assert res['s'] == 'file1.pdb'
    assert res['t'] == ['file2.yaml']
    assert res['o'] == 'file3.pdb'
    assert res['hydrogens']
    assert res['seed'] == 300


def test_backmap_cli_1(args_backmap):
    res = vars(cli.parse_cl_backmap(args_backmap[1]))
    assert res['d'] == 'database.db'
    assert res['s'] == 'file1.pdb'
    assert res['t'] == ['file2.yaml']
    assert res['o'] == 'file3.pdb'
    assert not res['hydrogens']
    assert res['seed'] is None


def test_backmap_cli_2(args_backmap):
    res = vars(cli.parse_cl_backmap(args_backmap[2]))
    assert res['d'] == 'database.db'
    assert res['s'] == 'file1.pdb'
    assert res['t'] == ['file2.yaml', 'file3.yaml', 'file4.yaml']
    assert res['o'] == 'file5.pdb'
    assert not res['hydrogens']
    assert res['seed'] is None


def test_backmap_cli_3(args_backmap):
    res = vars(cli.parse_cl_backmap(args_backmap[3]))
    assert res['d'] == 'database.db'
    assert res['s'] == 'file1.pdb'
    assert res['t'] == ['file2.yaml', 'file3.yaml', 'file4.yaml']
    assert res['o'] == 'file5.pdb'
    assert not res['hydrogens']
    assert res['seed'] is None


def test_backmap_cli_4(args_backmap):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cl_backmap(args_backmap[4])
    assert exc.value.code == 2


def test_backmap_cli_5(args_backmap):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cl_backmap(args_backmap[5])
    assert exc.value.code == 2


def test_mapping_cli_0(args_mapping):
    res = vars(cli.parse_cl_mapping(args_mapping[0]))
    assert res['a'] == 'file1.pdb'
    assert res['m'] == 'file2.pdb'
    assert res['o'] == 'file3.yaml'
    assert res['s'] == [['UND', 'CCCC']]


def test_mapping_cli_1(args_mapping):
    res = vars(cli.parse_cl_mapping(args_mapping[1]))
    assert res['a'] == 'file1.pdb'
    assert res['m'] == 'file2.pdb'
    assert res['o'] == 'file3.yaml'
    assert res['s'] == [['UND', 'CCCC'], ['PRO', 'CCCO']]


def test_mapping_cli_2(args_mapping):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cl_mapping(args_mapping[2])
    assert exc.value.code == 2


def test_mapping_cli_3(args_mapping):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cl_mapping(args_mapping[3])
    assert exc.value.code == 2


def test_cg_cli_0(args_cg):
    res = vars(cli.parse_cl_coarse_graining(args_cg[0]))
    assert res['a'] == 'file1.pdb'
    assert res['x'] == 'file2.xtc'
    assert res['t'] == ['file3.yaml']
    assert res['o'] == ['file4.pdb', 'file5.xtc']


def test_cg_cli_1(args_cg):
    res = vars(cli.parse_cl_coarse_graining(args_cg[1]))
    assert res['a'] == 'file1.pdb'
    assert res['t'] == ['file2.yaml']
    assert res['o'] == ['file3.pdb']


def test_cg_cli_2(args_cg):
    res = vars(cli.parse_cl_coarse_graining(args_cg[2]))
    assert res['a'] == 'file1.pdb'
    assert res['x'] == 'file2.xtc'
    assert res['t'] == ['file3.yaml', 'file4.yaml', 'file5.yaml']
    assert res['o'] == ['file6.pdb']


def test_cg_cli_3(args_cg):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cl_coarse_graining(args_cg[3])
    assert exc.value.code == 2


def test_posre_cli_0(args_posre):
    res = vars(cli.parse_cl_generate_posre(args_posre[0]))
    assert res['c'] == 'file1.pdb'
    assert res['t'] == ['file2.yaml']
    assert res['i'] == 'file3.itp'
    assert res['r'] == 'file4.pdb'


def test_posre_cli_1(args_posre):
    res = vars(cli.parse_cl_generate_posre(args_posre[1]))
    assert res['c'] == 'file1.pdb'
    assert res['t'] == ['file2.yaml', 'file3.yaml', 'file4.yaml']
    assert res['i'] == 'file5.itp'
    assert res['r'] == 'file6.pdb'


def test_posre_cli_2(args_posre):
    res = vars(cli.parse_cl_generate_posre(args_posre[2]))
    assert res['c'] == 'file1.pdb'
    assert res['t'] == ['file2.yaml']


def test_posre_cli_3(args_posre):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cl_generate_posre(args_posre[3])
    assert exc.value.code == 2
