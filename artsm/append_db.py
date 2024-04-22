import logging
import os
import sys

import numpy as np

from artsm.database.db import DBdata
from artsm.model.models import derive_models
from artsm.utils.other import setup_logger
from artsm.utils.config import parse_args_append, check_config_db, parse_simulations
from artsm.utils.cli import parse_cl_append
from artsm.utils.fileparsing import join_path


def get_target_db(args):
    logger = logging.getLogger(__name__)

    orig_db = DBdata(args.d, exist_ok=True)
    if args.o == args.d:  # append mode
        # data is appended to the existing db file using 'write_to_db' function,
        # if release is needed, original db file is deleted. The new changes are then present in the release file.
        logger.info(f'output database (-o) is the same as the source database (-d), so new data will be '
                    f'appended to the source database.')
        target_db = orig_db
    else:
        # create a copy of the database and then add new data to it using 'write_to_db' function
        # the original db file remains intact. Release file can be generated.
        if os.path.exists(args.o):
            logger.warning(f'output db file {args.o} already exists so it will be overwritten.')

        orig_db.copy_db(args.o)
        target_db = DBdata(args.o, exist_ok=True)
    return target_db


def main():
    """
    Append unseen fragment pairs to the database.
    """
    logger = setup_logger(__name__)

    logger.info('Read command line arguments.')
    args = parse_cl_append(sys.argv[1:])

    # set global seed
    rng = np.random.default_rng(args.seed)

    logger.info('Parse command line arguments.')
    config = parse_args_append(args)
    check_config_db(config)

    # returns a dict of Simulation objects
    simulations = parse_simulations(config, rng)

    # read data from release database
    logger.info('Read database.')

    target_db = get_target_db(args)
    logger.info('Reading data from the provided atomistic simulations')

    # the source database object
    database = DBdata(args.d, n_datapoints=args.n_datapoints, exist_ok=True)

    # get fr_pair_ids from the database and store as a list
    ignore_fr_pairs = database.get_fr_pair_ids() or []
    ignore_fr = database.get_fr_ids() or []

    for simulation_name, simulation in simulations.items():
        logger.info(f'Processing simulation {simulation_name}')

        if args.time_step is not None:
            logger.info(f'Setting time step to {args.time_step}')
            simulation.time_step = args.time_step

        logger.info('Derive topology of molecules and fragments.')
        simulation.derive_topology()

        logger.info('Read simulation data.')
        simulation.read_simulation(memory=False)

        logger.info('Extract bond and angle values.')
        simulation.extract_bond_data()
        simulation.extract_angle_data()

        logger.info('Extract data for main conformations, fragment pairs, and machine learning models from simulation.')
        simulation.extract_fr_data()

        logger.info('Appending new data to the output database object.')
        simulation.write_to_db(target_db, ignore_fr_pairs, ignore_fr)

    # derive models for new fragment pairs and fragments
    logger.info('Derive models for new fragment pairs and fragments.')
    derive_models(target_db, os.path.dirname(args.o), rng, args.seed, ignore_fr_pairs, ignore_fr)

    if args.release:
        filename = join_path(args.o, 'release.db')
        target_db.copy_db(filename, release=True)
    logger.info('Database object has been saved to disk.')


if __name__ == '__main__':
    main()
