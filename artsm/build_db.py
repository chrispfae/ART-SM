#!/usr/bin/env python3
import os
import sys

import numpy as np

from artsm.database.db import DBdata
from artsm.model.models import derive_models
from artsm.utils.cli import parse_cl_db
from artsm.utils.config import parse_args, check_config_db, parse_simulations
from artsm.utils.fileparsing import join_path
from artsm.utils.other import setup_logger


def main():
    """
    Main function for the database building tool.
    """
    logger = setup_logger(__name__)
    logger.info('Read command line arguments.')
    args = parse_cl_db(sys.argv[1:])

    # set global seed
    rng = np.random.default_rng(args.seed)

    logger.info('Parse command line arguments.')
    config = parse_args(args)
    check_config_db(config)
    simulations = parse_simulations(config, rng)

    logger.info('Start building a fragment data base from atomistic simulations.')

    logger.info('Initialize empty database.')
    database = DBdata(args.d, n_datapoints=args.n_datapoints, delete=True)

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

        logger.info('Write fragments and fragment pairs to database.')
        simulation.write_to_db(database)

    logger.info('Determine model for all fragment pairs and one bead molecules.')
    derive_models(database, os.path.dirname(args.d), rng, args.seed)

    logger.info('Generate final database.')
    filename = join_path(args.d, 'release.db')
    database.release(filename, delete=True)


if __name__ == '__main__':
    main()
