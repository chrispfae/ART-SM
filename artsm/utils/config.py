import copy
import os.path
import sys

from artsm.topology.simulation import parse_snapshot, Simulation
from artsm.utils.containers import check_keys, remove_keys
from artsm.utils.fileparsing import join_path, read_yaml
from artsm.utils.other import setup_logger
from artsm.water.data import supported_water_models


def parse_args(args):
    """
    Parse the command line arguments, read specified files, and return a configuration dictionary.

    The configuration dictionary is created based on the provided arguments.
    - If the 'g' arguments is provided, the global configuration file is parsed
    - If the 'c' arguments is provided, the configuration file for a single simulation is parsed
    - If neither 'g' nor 'c' is provided, the provided arguments (like 's', 't', 'x')
      are parsed to create the configuration dictionary.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    dict
        A configuration dictionary based on the provided arguments.
    """
    if hasattr(args, 'g') and args.g is not None:
        config = parse_config_g(args.g)
    elif hasattr(args, 'c') and args.c is not None:
        config = {'sim1': parse_config_sim(args.c)}
    else:
        relevant_keys = ['o', 's', 't', 'x', 'time_step']
        config_sim = vars(args)
        config_sim_new = {key_: value_ for key_, value_ in config_sim.items() if key_ in relevant_keys and value_ is not None}
        modify_stx(config_sim_new)
        config = {'sim1': config_sim_new}

    # Ensure that each simulation config has the time_step key if building a database
    if hasattr(args, 'time_step'):
        for config_sim in config.values():
            if 'time_step' not in config_sim:
                config_sim['time_step'] = args.time_step
    return config


def parse_args_append(args):
    """
    Parse the command line arguments, read specified files, and return a configuration dictionary for append_db.

    The configuration dictionary is created based on the provided arguments.
    - If the 'g' arguments is provided, the global configuration file is parsed
    - If the 'c' arguments is provided, the configuration file for a single simulation is parsed
    - If neither 'g' nor 'c' is provided, the provided arguments (like 's', 't', 'x')
      are parsed to create the configuration dictionary.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    dict
        A configuration dictionary based on the provided arguments.
    """
    if hasattr(args, 'g') and args.g is not None:
        config = parse_config_g(args.g)
    elif hasattr(args, 'c') and args.c is not None:
        config = {'sim1': parse_config_sim(args.c)}
    else:
        relevant_keys = ['d', 's', 't', 'x', 'time_step']
        config_sim = vars(args)
        config_sim_new = {key_: value_ for key_, value_ in config_sim.items() if key_ in relevant_keys and value_ is not None}
        modify_stx(config_sim_new)
        config = {'sim1': config_sim_new}

    # Ensure that each simulation config has the time_step key if building a database
    if hasattr(args, 'time_step'):
        for config_sim in config.values():
            if 'time_step' not in config_sim:
                config_sim['time_step'] = args.time_step
    return config     


def parse_config_g(filename):
    """
    Parse the global configuration file (yaml format) and return a dictionary of simulation configurations.

    Parameters
    ----------
    filename : str
        The path to the global configuration file.

    Returns
    -------
    dict
        The parsed configuration settings. The keys are simulation names and the values are the parsed
        configuration dictionaries for each simulation.
    """
    config_g = read_yaml(filename)
    for sim_name, sim_filename in config_g.items():
        config_g[sim_name] = parse_config_sim(join_path(filename, sim_filename))
    return config_g


def parse_config_sim(filename):
    """
    Parse a simulation config file (yaml format) and return a dictionary of the simulations' configuration.

    Parameters
    ----------
    filename : str
        The path to the configuration file.

    Returns
    -------
        dict: The parsed configuration settings.
    """
    config_sim = read_yaml(filename)
    modify_stx(config_sim, filename)
    return config_sim


def modify_stx(config, path=None):
    """
    Modify the values of the keys s, t, and x.

    - 's': Update path.
    - 't': Read the specified yaml file and store resulting dict.
    - 'x': Update path.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    path : str, default None
        The current working directory. Default value is transformed to '.'.
    """
    if path is None:
        path = '.'
    if 's' in config:
        s_file_path = join_path(path, config['s'])
        config['s'] = s_file_path
    if 't' in config:
        # t has to be either a list or string containing paths to mapping files
        t = config['t']
        if isinstance(t, str):
            t = t.split()
        elif not isinstance(t, list):
            logger = setup_logger(__name__)
            logger.error(f'Argument t {t} neither a list nor a string. Abort...')
            sys.exit(-1)

        unique_opts = list(set(t))
        if len(unique_opts) > 1:
            merge_dict = {}
            for topo in unique_opts:
                t_file_path = join_path(path, topo)
                from_yaml = read_yaml(t_file_path)
                common_keys = set(merge_dict.keys()).intersection(set(from_yaml.keys()))
                if len(common_keys) > 0:
                    logger = setup_logger(__name__)
                    logger.warning(f"found redundant entries for {common_keys} in {t_file_path}")
                merge_dict.update(from_yaml)
            config['t'] = merge_dict
        else:
            t_file_path = join_path(path, unique_opts[0])
            config['t'] = read_yaml(t_file_path)
    if 'x' in config:
        x_file_path = join_path(path, config['x'])
        config['x'] = x_file_path


def check_config_db(config):
    """
    Check the validity of the provided configuration for the database creation process.

    Parameters
    ----------
    config : dict
        The configuration dictionary.

    Raises
    ------
        Error: If any of the provided options is invalid.
    """
    for config_sim in config.values():
        required_keys = ['s', 'x', 't']
        check_keys(config_sim, required_keys)
        remove_keys(config_sim, required_keys + ['time_step'])
        for config_mol in config_sim['t'].values():
            if isinstance(config_mol, str):
                if config_mol not in supported_water_models:
                    logger = setup_logger(__name__)
                    logger.error(f'The specified Water model {config_mol} is not available. '
                                 f'However, the following water models are supported: {supported_water_models.keys()}')
                    sys.exit(-1)
            else:
                required_keys = ['smiles', 'adj_atoms', 'mapping', 'charges']
                check_keys(config_mol, required_keys)
                remove_keys(config_mol, required_keys)


def check_config_bm(config):
    """
    Check the validity of the provided configuration for backmapping.

    Parameters
    ----------
    config : dict
        The configuration dictionary.

    Raises
    ------
        Error: If any of the provided options is invalid.
    """
    for config_sim in config.values():
        required_keys = ['o', 's', 't']
        check_keys(config_sim, required_keys)
        remove_keys(config_sim, required_keys)
        for config_mol in config_sim['t'].values():
            if isinstance(config_mol, str):
                if config_mol not in supported_water_models:
                    logger = setup_logger(__name__)
                    logger.error(f'The specified Water model {config_mol} is not available. '
                                 f'However, the following water models are supported: {supported_water_models.keys()}')
                    sys.exit(-1)
            else:
                required_keys = ['smiles', 'adj_atoms', 'mapping', 'charges']
                check_keys(config_mol, required_keys)
                valid_keys = ['smiles', 'adj_atoms', 'mapping', 'charges', 'atom_order']
                remove_keys(config_mol, valid_keys)


def parse_simulations(config, rng):
    """
    Parse the simulation config files and return a dictionary of Simulation objects.

    Parameters
    ----------
    config: dict
        Dictionary containing as keys the simulation names and as values the simulation configurations.
    rng: np.random.default_rng()
        Default random number generator of numpy.

    Returns
    -------
    dict
        Dictionary containing the as keys the simulation names and as values the Simulation objects.
    """
    config_parsed = copy.deepcopy(config)

    for sim_name, sim_config in config_parsed.items():
        if 'o' in sim_config:
            del sim_config['o']
        config_parsed[sim_name] = Simulation(**sim_config, rng=rng)
    return config_parsed


# def parse_snapshot_config(filenames):
#     """
#     Parses the snapshot configuration from a YAML file.
#
#     It is ensured that the 'path' key is present in the returned dictionary. The resulting dictionary
#     is passed to the `parse_snapshot` function to be parsed.
#
#     Args:
#         filenames (str): The path to the YAML file.
#
#     Returns:
#         Simulation object: A Simulation object created from the parsed YAML file.
#     """
#     config = read_yaml(filenames)
#     check_key_in_dict(config, 'path', filenames)
#     config['path'] = filenames
#     return parse_snapshot(config)
