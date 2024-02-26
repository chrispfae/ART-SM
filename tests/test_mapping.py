import sys

from artsm.mapping import main
from artsm.utils.fileparsing import read_yaml
from fixtures import args_mapping_module, file_mapping0


def test_mapping0(args_mapping_module, file_mapping0):
    sys.argv = args_mapping_module[0]
    main()
    res = read_yaml(args_mapping_module[1])
    # Water
    assert res['TIP'] == 'TIP3P'
    # PRP
    assert sorted(list(res['PRP'].keys())) == ['adj_atoms', 'atom_order', 'mapping', 'smiles']
    assert res['PRP']['adj_atoms'] == {'H1': ['C1'], 'C1': ['H1', 'H2', 'H3', 'C2'], 'H2': ['C1'], 'H3': ['C1'],
                                       'C2': ['C1', 'H4', 'H5', 'C3'], 'H4': ['C2'], 'H5': ['C2'],
                                       'C3': ['C2', 'H6', 'H7', 'O'], 'H6': ['C3'], 'H7': ['C3'], 'O': ['C3', 'H8'],
                                       'H8': ['O']}
    assert res['PRP']['atom_order'] == ['H1', 'C1', 'H2', 'H3', 'C2', 'H4', 'H5', 'C3', 'H6', 'H7', 'O', 'H8']
    assert res['PRP']['mapping']['C1A'] == ['H1', 'C1', 'H2', 'H3', 'C2', 'H4', 'H5', 'C3', 'H6', 'H7', 'O', 'H8']
    assert res['PRP']['smiles'] == 'CCCO'
    # HEP
    assert sorted(list(res['HEP'].keys())) == ['adj_atoms', 'atom_order', 'mapping', 'smiles']
    assert res['HEP']['adj_atoms'] == {'O': ['C1', 'H16'], 'H16': ['O'], 'C1': ['O', 'H1', 'H2', 'C2'], 'H1': ['C1'],
                                       'H2': ['C1'], 'C2': ['C1', 'H3', 'H4', 'C3'], 'H3': ['C2'], 'H4': ['C2'],
                                       'C3': ['C2', 'H5', 'H6', 'C4'], 'H5': ['C3'], 'H6': ['C3'],
                                       'C4': ['C3', 'H7', 'H8', 'C5'], 'H7': ['C4'], 'H8': ['C4'],
                                       'C5': ['C4', 'H9', 'H10', 'C6'], 'H9': ['C5'], 'H10': ['C5'],
                                       'C6': ['C5', 'H11', 'H12', 'C7'], 'H11': ['C6'], 'H12': ['C6'],
                                       'C7': ['C6', 'H13', 'H14', 'H15'], 'H13': ['C7'], 'H14': ['C7'], 'H15': ['C7']}
    assert res['HEP']['atom_order'] == ['O', 'C1', 'H1', 'H2', 'C2', 'H3', 'H4', 'C3', 'H5', 'H6', 'C4', 'H7', 'H8',
                                        'C5', 'H9', 'H10', 'C6', 'H11', 'H12', 'C7', 'H13', 'H14', 'H15', 'H16']
    assert res['HEP']['mapping']['C1O'] == ['O', 'C1', 'H1', 'H2', 'C2', 'H3', 'H4', 'C3', 'H5', 'H6', 'H16']
    assert res['HEP']['mapping']['C2A'] == ['C4', 'H7', 'H8', 'C5', 'H9', 'H10', 'C6', 'H11', 'H12', 'C7', 'H13',
                                            'H14', 'H15']
    assert res['HEP']['smiles'] == 'CCCCCCCO'
    # UND
    assert sorted(list(res['UND'].keys())) == ['adj_atoms', 'atom_order', 'mapping', 'smiles']
    assert res['UND']['adj_atoms'] == {'O': ['C1', 'H24'], 'H24': ['O'], 'C1': ['O', 'H1', 'H2', 'C2'], 'H1': ['C1'],
                                       'H2': ['C1'], 'C2': ['C1', 'H3', 'H4', 'C3'], 'H3': ['C2'], 'H4': ['C2'],
                                       'C3': ['C2', 'H5', 'H6', 'C4'], 'H5': ['C3'], 'H6': ['C3'],
                                       'C4': ['C3', 'H7', 'H8', 'C5'], 'H7': ['C4'], 'H8': ['C4'],
                                       'C5': ['C4', 'H9', 'H10', 'C6'], 'H9': ['C5'], 'H10': ['C5'],
                                       'C6': ['C5', 'H11', 'H12', 'C7'], 'H11': ['C6'], 'H12': ['C6'],
                                       'C7': ['C6', 'H13', 'H14', 'C8'], 'H13': ['C7'], 'H14': ['C7'],
                                       'C8': ['C7', 'H15', 'H16', 'C9'], 'H15': ['C8'], 'H16': ['C8'],
                                       'C9': ['C8', 'H17', 'H18', 'C10'], 'H17': ['C9'], 'H18': ['C9'],
                                       'C10': ['C9', 'H19', 'H20', 'C11'], 'H19': ['C10'], 'H20': ['C10'],
                                       'C11': ['C10', 'H21', 'H22', 'H23'], 'H21': ['C11'], 'H22': ['C11'],
                                       'H23': ['C11']}
    assert res['UND']['atom_order'] == ['O', 'C1', 'H1', 'H2', 'C2', 'H3', 'H4', 'C3', 'H5', 'H6', 'C4', 'H7', 'H8',
                                        'C5', 'H9', 'H10', 'C6', 'H11', 'H12', 'C7', 'H13', 'H14', 'C8', 'H15', 'H16',
                                        'C9', 'H17', 'H18', 'C10', 'H19', 'H20', 'C11', 'H21', 'H22', 'H23', 'H24']
    assert sorted(res['UND']['mapping']['C1O']) == sorted(['O', 'C1', 'H1', 'H2', 'C2', 'H3', 'H4', 'C3', 'H5', 'H6',
                                                           'H24'])
    assert sorted(res['UND']['mapping']['C2A']) == sorted(['C4', 'H7', 'H8', 'C5', 'H9', 'H10', 'C6', 'H11', 'H12',
                                                           'C7', 'H13', 'H14'])
    assert sorted(res['UND']['mapping']['C3A']) == sorted(['C8', 'H15', 'H16', 'C9', 'H17', 'H18', 'C10', 'H19', 'H20',
                                                           'C11', 'H21', 'H22', 'H23'])
    assert res['UND']['smiles'] == 'CCCCCCCCCCCO'


def test_mapping1(args_mapping_module, file_mapping0):
    sys.argv = args_mapping_module[2]
    main()
    res = read_yaml(args_mapping_module[1])
    # Water
    assert res['TIP'] == 'TIP4P'
    