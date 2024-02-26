from artsm.utils.config import parse_args, check_config_db, check_config_bm
import artsm.utils.cli as cli
from fixtures import args_build_db_data, args_backmap_data


def test_build_db_argparse_0(args_build_db_data):
    args = cli.parse_cl_db(args_build_db_data[0])
    res = parse_args(args)
    check_config_db(res)
    res = res['sim1']
    assert res['s'] == 'data/snapshots/atomistic.pdb'
    assert res['x'] == 'data/snapshots/atomistic.xtc'
    assert res['time_step'] == 400
    assert sorted(list(res['t'].keys())) == ['HEP', 'PRP', 'TIP', 'UND']


def test_build_db_argparse_1(args_build_db_data):
    args = cli.parse_cl_db(args_build_db_data[1])
    res = parse_args(args)
    check_config_db(res)
    res = res['sim1']
    assert res['s'] == 'data/snapshots/atomistic.pdb'
    assert res['x'] == 'data/snapshots/atomistic.xtc'
    assert res['time_step'] == 500
    assert sorted(list(res['t'].keys())) == ['HEP', 'PRP', 'TIP', 'UND']


def test_build_db_argparse_2(args_build_db_data):
    args = cli.parse_cl_db(args_build_db_data[2])
    res = parse_args(args)
    check_config_db(res)
    res = res['sim1']
    assert res['s'] == 'data/snapshots/atomistic.pdb'
    assert res['x'] == 'data/snapshots/atomistic.xtc'
    assert res['time_step'] == 500
    assert sorted(list(res['t'].keys())) == ['HEP', 'PRP', 'TIP', 'UND']


def test_build_db_argparse_3(args_build_db_data):
    args = cli.parse_cl_db(args_build_db_data[3])
    res = parse_args(args)
    check_config_db(res)
    res = res['sim0']
    assert res['s'] == 'data/snapshots/atomistic.pdb'
    assert res['x'] == 'data/snapshots/atomistic.xtc'
    assert res['time_step'] == 500
    assert sorted(list(res['t'].keys())) == ['HEP', 'PRP', 'TIP', 'UND']


def test_build_db_argparse_4(args_build_db_data):
    args = cli.parse_cl_db(args_build_db_data[4])
    res = parse_args(args)
    check_config_db(res)
    res1 = res['sim0']
    assert res1['s'] == 'data/snapshots/atomistic.pdb'
    assert res1['x'] == 'data/snapshots/atomistic.xtc'
    assert res1['time_step'] == 200
    assert sorted(list(res1['t'].keys())) == ['HEP', 'PRP', 'TIP', 'UND']
    res2 = res['sim1']
    assert res2['s'] == 'data/snapshots/atomistic.pdb'
    assert res2['x'] == 'data/snapshots/atomistic.xtc'
    assert res2['time_step'] == 1100
    assert sorted(list(res2['t'].keys())) == ['HEP', 'PRP', 'TIP', 'UND']


def test_build_db_argparse_5(args_build_db_data):
    args = cli.parse_cl_db(args_build_db_data[5])
    res = parse_args(args)
    assert 'random' in res['sim1']
    check_config_db(res)
    res = res['sim1']
    assert 'random' not in res
    assert res['s'] == 'data/snapshots/atomistic.pdb'
    assert res['x'] == 'data/snapshots/atomistic.xtc'
    assert sorted(list(res['t'].keys())) == ['HEP', 'PRP', 'TIP', 'UND']


def test_backmap_argparse_0(args_backmap_data):
    args = cli.parse_cl_backmap(args_backmap_data[0])
    res = parse_args(args)
    check_config_bm(res)
    res = res['sim1']
    assert res['s'] == 'data/snapshots/cg.pdb'
    assert res['o'] == 'backmapped.pdb'
    assert sorted(list(res['t'].keys())) == ['HEP', 'PRP', 'TIP', 'UND']
