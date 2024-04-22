import argparse
import sys

from artsm.utils.other import setup_logger


def parse_common_args(parser):
    """
    Adds common command-line arguments for building or appending a database to an ArgumentParser.

    Parameters:
    ----------
    parser : argparse.Namespace
        Parser that is extended.

    Returns
    -------
    parser : argparse.Namespace
        Extended parser.
    """
    group = parser.add_argument_group('config files')
    exclusive_group = group.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument('-c', '--config_simulation', type=str, dest='c',
                                 help='Input simulation config file that should only specify the arguments -s, -t, -x, '
                                      'and optionally --time_step in yaml format.'
                                      'Use this argument for a single simulation only. '
                                      'Alternatively, you can directly provide these arguments on the command line.')
    exclusive_group.add_argument('-g', '--global_config', type=str, dest='g',
                                 help='Input global config file that contains paths to simulation config files. See argument -c.'
                                      'Use this argument for multiple simulations.')
    parser.add_argument('-s', '--snapshot', nargs='?', type=str, dest='s',
                        help='Input atomistic snapshot, e.g. pdb file.')
    parser.add_argument('-t', '--topology', nargs='*', type=str, dest='t', action='extend',  # requires python 3.8
                        help='Input mapping file that contains topology of molecules and mapping to coarse-grained resolution.')
    parser.add_argument('-x', '--xtc', nargs='?', type=str, dest='x', help='Input atomistic trajectory, e.g. xtc file.')
    parser.add_argument('--time_step', nargs='?', type=int, default=500, const=500,
                        help='Snapshots are used only every [--time_step] ps to build the database.'
                             'Ensures that independent datapoints are extracted from the simulations.')
    parser.add_argument('--seed', nargs='?', type=int,
                        help='Set the seed to obtain reproducible results.')
    parser.add_argument('--n_datapoints', nargs='?', type=int, default=500, const=500,
                        help='Maximum number of datapoints of fragment-pairs used for training the ML model. '
                             'Increasing the number of datapoints will significantly increase the runtime.')
    return parser


def check_input_args(args):
    """
    Performs checks on parsed arguments to ensure valid input combinations for building or appending a database.
    If necessary the arguments are modified to a common format.

    Parameters
    ----------
    args : argparse.Namespace
        Parser to be checked.

    Returns
    -------
    argparse.Namespace
        The validated parser.
    """
    args_mod = [key_ for key_, value_ in vars(args).items() if value_ is not None]
    if ('g' in args_mod or 'c' in args_mod) and len(args_mod) > 1:
        for arg in args_mod:
            if arg in ['s', 't', 'x']:
                logger = setup_logger(__name__)
                logger.warning(f'Option -g is provided. Additional argument -{arg} is ignored. '
                               f'Please provide it in the individual simulation config files.')
                setattr(args, arg, None)
    else:
        for arg in ['s', 'x', 't']:
            if arg not in args_mod:
                logger = setup_logger(__name__)
                logger.error(f'Option -{arg} is missing. Either -g or -c or (-s, -t, -x) have to be provided.')
                sys.exit(-1)
    return args


def parse_cl_db(cl):
    """
    Parse command line arguments for building a fragment-pair database.

    Parameters:
    ----------
    cl : list of str
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(prog='artsm-build_db',
                                     description='Building a database of fragment-pairs from atomistic simulations. '
                                                 'Minimum requirements are an atomistic snapshot (e.g. pdb), '
                                                 'trajectory (e.g. xtc) and a mapping file that indicates the mapping '
                                                 'from atomistic to coarse-grained resolution. For a single simulation they can be'
                                                 'provided with the arguments -s, -x, and -t or with a simulation '
                                                 'config file (argument -c). For multiple simulations please specify a '
                                                 'global config file with the argument -g.')
    parser = parse_common_args(parser)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-d', '--database', type=str, dest='d', required=True,
                               help='Output database file for storing fragments, fragment-pairs and their models.')

    args = parser.parse_args(cl)

    # Multiple possibilities to provide input data. Thus, we need checks in addition to argparse.
    args = check_input_args(args)
    return args


def parse_cl_append(cl):
    parser = argparse.ArgumentParser(prog='artsm-append',
                                     description='Appending an existing database with new fragments and fragment pairs.' 
                                                 'Required are the orgignal database -d and the output database -o. '
                                                 'Moreover, an atomistic snapshot (e.g. pdb), '
                                                 'trajectory (e.g. xtc) and a mapping file that indicates the mapping '
                                                 'from atomistic to coarse-grained resolution is mandatory. '
                                                 'For a single simulation they can be provided with the arguments '
                                                 '-s, -x, and -t or with a simulation config file (argument -c). '
                                                 'For multiple simulations please specify a global config file with '
                                                 'the argument -g.')
    parser = parse_common_args(parser)
    # Specific arguments for parse_cl_append
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-d', '--database', type=str, dest='d', required=True,
                                 help='Existing database that will be appended.')
    requiredNamed.add_argument('-o', '--output', type=str, dest='o', required=True,
                               help='Output database that includes the appended data.')
    parser.add_argument('--release', action='store_true', help='Additional release database is generated, which only'
                                                               'contains the derived model without the '
                                                               'training / atomistic data.')
    args = parser.parse_args(cl)

    # Multiple possibilities to provide input data. Thus, we need checks in addition to argparse.
    args = check_input_args(args)
    return args


def parse_cl_backmap(cl):
    """
    Parse command line arguments for backmapping a coarse-grained structure to atomistic resolution.

    Parameters
    ----------
    cl : list of str
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(prog='artsm-backmap',
                                     description='Backmapping of a coarse-grained structure to atomistic resolution.'
                                                 'Minimum requirements are a coarse-grained snapshot -s (e.g. pdb), '
                                                 'a database of fragment-pairs -d (generate with artsm-build_db),'
                                                 'and a mapping file -t that indicates the mapping '
                                                 'from atomistic to coarse-grained resolution. '
                                                 'The output is specified with -o.')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-s', '--snapshot', type=str, dest='s', required=True,
                               help='Input Coarse-grained snapshot, e.g. pdb file.')
    requiredNamed.add_argument('-d', '--database', type=str, dest='d', required=True,
                               help='Input database of fragment-pairs.')
    requiredNamed.add_argument('-t', '--topology', nargs='*', type=str, dest='t', action='extend', required=True,
                               help='Input mapping file that contains topology '
                                    'of molecules and mapping to coarse-grained resolution.')
    requiredNamed.add_argument('-o', '--output', type=str, dest='o', required=True,
                               help='Output pdb file of the backmapped structure.')
    parser.add_argument('--hydrogens', action='store_true', help='Add hydrogen atoms to the backmapped structures.')
    parser.add_argument('--seed', nargs='?', type=int,
                        help='Set the seed to obtain reproducible results.')
    args = parser.parse_args(cl)

    return args


def parse_cl_mapping(cl):
    """
    Parse command line arguments for generating a mapping file.

    Parameters
    ----------
    cl : list of str
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(prog='artsm-mapping',
                                     description='Generate a mapping file, which can be used to generate a database of '
                                                 'fragment-pairs and for backmapping. '
                                                 'Required are a matching atomistic and coarse-grained snapshot, '
                                                 'which can contain single or multiple molecules.'
                                                 'For all molecules the correspondig SMILES '
                                                 'representation has to be specified.')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-a', type=str, required=True, help='Input atomistic snapshot, e.g. pdb.')
    requiredNamed.add_argument('-m', type=str, required=True, help='Input corresponding coarse-grained model.')
    requiredNamed.add_argument('-s', '--smiles', nargs=2, action='append', type=str, dest='s', required=True,
                               help='Smiles representation of respective molecules. '
                                    'Provide this argument multiple times for the different molecules in the snapshots,'
                                    'e.g. -s PRO CCCC -s UND CCCCCCCCO. Please also provide the SMILES for '
                                    'water in your system with \'-s ID O\'')
    requiredNamed.add_argument('-o', type=str, required=True, help='Output mapping file in yaml format.')
    parser.add_argument('-w', '--water', nargs='?', default='TIP3P', const='TIP3P', dest='w',
                        help='Specify which water model you want to use. Default is TIP3P.')
    return parser.parse_args(cl)


def parse_cl_coarse_graining(cl):
    """
    Parse command line arguments for coarse-graining an atomistic structure.

    Parameters
    ----------
    cl : list of str
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(prog='artsm-coarse_grain',
                                     description='Manual coarse-graining of an atomistic structure. '
                                                 'Required are the atomistic structure -a and a mapping file -t.')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-a', type=str, required=True, help='Input atomistic snapshot, e.g. pdb file.')
    parser.add_argument('-x', nargs='?', type=str, help='Input atomistic trajectory, e.g. xtc file.')
    requiredNamed.add_argument('-t', '--topology', nargs='*', type=str, action='extend', dest='t', required=True,
                               help='Input mapping file that contains topology of molecules and mapping to coarse-grained '
                                    'resolution.')
    requiredNamed.add_argument('-o', nargs='*', type=str, action='extend', required=True,
                               help='Output coarse-grained snapshot (and trajectory). '
                                    'First argument is the snapshot (and the second the trajectory).'
                                    'For instance, \'-o cg.pdb cg.xtc\' will output a snapshot and trajectory if -x.'
                                    '\'-o cg.pdb\' will only output the snapshot')
    return parser.parse_args(cl)


def parse_cl_generate_posre(cl):
    """
    Parse command line arguments for generating position restraint itp files.

    Parameters
    ----------
    cl : list of str
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(prog='artsm-posre',
                                     description='Generate flat-bottomed position restraint files on heavy atoms '
                                                 'for relaxation after backmapping.'
                                                 'Required are the coarse-grained file that has been backmapped -c '
                                                 'and the corresponding mapping file -t')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-c', type=str, required=True, help='Input coarse-grained structure, e.g. pdb file.')
    requiredNamed.add_argument('-t', '--topology', nargs='*', type=str, action='extend', dest='t', required=True,
                               help='Input mapping file that contains topology of molecules and mapping to coarse-grained '
                                    'resolution.')
    parser.add_argument('-i', nargs='?', type=str,
                        help='Output itp file. Prefix is automatically added for different molecules.')
    parser.add_argument('-r', nargs='?', type=str,
                        help='Output structure file for position restraint simulation, e.g. gro file.')
    return parser.parse_args(cl)
