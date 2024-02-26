import os
import sys

import numpy as np

from artsm.database.db import DBdata
from artsm.utils.cli import parse_cl_backmap
from artsm.utils.config import parse_args, check_config_bm, parse_simulations
from artsm.utils.other import setup_logger

import warnings


def main():
    """
    Main function for the backmapping tool.
    """
    logger = setup_logger(__name__)
    logger.info('Read command line arguments.')
    args = parse_cl_backmap(sys.argv[1:])

    # set global seed
    rng = np.random.default_rng(args.seed)

    logger.info('Parse command line arguments.')
    config = parse_args(args)
    check_config_bm(config)
    simulations = parse_simulations(config, rng)  # config->simulations->molecules->fragments

    logger.info('Read database.')
    database = DBdata(args.d, exist_ok=True)

    for simulation_name, snapshot in simulations.items():
        logger.info(f'Processing simulation {simulation_name}')
        logger.info('Derive topology of molecules and fragments.')
        snapshot.derive_topology()

        logger.info('Load models from database.')
        snapshot.load_models_db(database)

        logger.info('Read snapshot data.')
        snapshot.read_simulation(memory=False)

        logger.info('Perform backmapping.')
        atomistic = snapshot.backmap(database, rng, args.hydrogens)

        logger.info('Write backmapped structure to file.')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            atomistic.write(config[simulation_name]['o'])


if __name__ == '__main__':
    main()

