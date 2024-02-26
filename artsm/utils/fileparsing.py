import logging
import os
import sys

import yaml

from artsm.utils.other import setup_logger


def read_yaml(filename):
    """
    Read yaml file and return its content.

    Parameters
    ----------
    filename : str
        The path to the yaml file.

    Returns
    -------
    dict
        The content of the yaml file.

    Raises
    ------
    yaml.YAMLError: If the yaml file cannot be parsed.
    """
    with open(filename, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger = setup_logger(__name__)
            logger.error(exc)
            sys.exit(-1)


def write_yaml(filename, *data):
    """
    Write dictionaries to yaml file.

    Multiple dictionaries can be provided.

    Parameters
    ----------
    filename : str
        The path to the output yaml file.
    *data : tuple of dict
        Dictionaries that are written to the yaml file.

    Raises
    ------
        yaml.YAMLError: If the yaml file cannot be written.
    """
    with open(filename, 'w') as stream:
        try:
            for data_block in data:
                yaml.dump(data_block, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            logger = setup_logger(__name__)
            logger.error(exc)
            sys.exit(-1)


def join_path(path, relative_path):
    """
    Join two paths, a base path and a corresponding relative path.

    Parameters
    ----------
    path : str
        The base path.
    relative_path : str
        The relative path.

    Returns
    -------
    str
        The joined path.
    """
    return os.path.normpath(os.path.join(os.path.dirname(path), relative_path))


def write_smiles(filename, smiles):
    with open(filename, 'w') as outfile:
        outfile.write(smiles)
