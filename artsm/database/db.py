import os
import pickle
import re
import sqlite3
import sys

import numpy as np

from artsm.topology.fragment import Fragment
from artsm.topology.fragment_pair import FragmentPair, Connector
from artsm.utils.other import deserialize, serialize, setup_logger


def combine_data(db_entry, data_fr_pair):
    """
    Combine existing data in the database with new data from a fragment pair.

    Parameters
    ----------
        db_entry: list serialized.
            The database entry containing the serialized data.
        data_fr_pair: list
            Fragment pair data.
    Returns
    -------
        The combined serialized data.
    """
    if db_entry is not None:
        data_db = deserialize(*db_entry)
        if len(data_db[0]) == 0:
            data_new = serialize(*data_fr_pair)
        else:
            data_new = [np.concatenate((i, j)) for i, j in zip(data_db, data_fr_pair)]

            data_new = serialize(*data_new)
    else:
        data_new = serialize(*data_fr_pair)

    return data_new


class DBdata:
    def __init__(self, filename, n_datapoints=500, exist_ok=False, delete=False):
        """
        SQLite3 database for storing fragments, fragment-pairs, and their corresponding data. Six tables are created:
        fr_pairs, fragments, data_fr_pairs, data_frs, bonds, and angles.
        
        Parameters
        ----------
        filename : str
            File that stores the database on disk.
        exist_ok : bool, default False
            If False, a FileExistsError is raised if the database file already exists to prevent overwriting.
        delete : bool, default False
            If True, the database is deleted if it already exists. Is processed before 'exist_ok',
            i.e. the database is deleted even if 'exist_ok' is False.
        """
        self.filename = filename
        self.n_datapoints = n_datapoints
        if delete and os.path.isfile(self.filename):
            os.remove(self.filename)

        if not exist_ok and os.path.isfile(self.filename):
            logger = setup_logger(__name__)
            logger.error(f'Database \'{self.filename}\' already exists.')
            raise FileExistsError

        try:
            self._connection = sqlite3.connect(self.filename)
            self._cursor = self._connection.cursor()
        except Exception as err:
            logger = setup_logger(__name__)
            logger.error(f'Could not connect to database \'{self.filename}\' due to error: {err}.')
            raise

        sqlite3.register_adapter(np.int_, lambda val: int(val))  # Enables to add variables with np.int64 type.

        # table stores information about the fragments
        self._cursor.execute('''CREATE TABLE if not exists fragments (fr_id INTEGER PRIMARY KEY, smiles TEXT NOT NULL,
                             atoms BLOB NOT NULL, A BLOB NOT NULL, internal_coords BLOB NOT NULL, main_coords BLOB, 
                             model BLOB)''')
        # table stores information about the fr_pair between two fragments
        self._cursor.execute('''CREATE TABLE if not exists fr_pairs (id INTEGER PRIMARY KEY, fr1_id INTEGER NOT NULL,
                             fr2_id INTEGER NOT NULL, smiles TEXT NOT NULL, atoms BLOB NOT NULL, elements BLOB NOT NULL,
                             bond_types BLOB NOT NULL, main_coords1 BLOB, main_internal BLOB, 
                             main_coords2 BLOB, main_internal2 BLOB, main_dihedrals BLOB, model BLOB)''')

        self._cursor.execute('''CREATE TABLE if not exists data_fr_pairs (id INTEGER PRIMARY KEY, zmatrix1 BLOB NOT NULL,
                             coords1 BLOB NOT NULL, zmatrix2 BLOB NOT NULL, coords2 BLOB NOT NULL, X BLOB NOT NULL, 
                             dihedral BLOB NOT NULL)''')
        self._cursor.execute('''CREATE TABLE if not exists data_frs (fr_id INTEGER PRIMARY KEY, zmatrix BLOB NOT NULL,
                                     coords BLOB NOT NULL)''')

        self._cursor.execute('''CREATE TABLE if not exists bonds (element1 TEXT NOT NULL, element2 TEXT NOT NULL, 
                             bond_type INTEGER NOT NULL, val REAL NOT NULL, datapoints INTEGER NOT NULl)''')
        self._cursor.execute('''CREATE TABLE if not exists angles (element1 TEXT NOT NULL, element2 TEXT NOT NULL, 
                             element3 TEXT NOT NULL, bond_type1 INTEGER NOT NULL, bond_type2 INTEGER NOT NULL,
                             val REAL NOT NULL, datapoints INTEGER NOT NULl)''')

        self._connection.commit()

    def print(self, data=True):
        """
        Print selected information from various tables in the database.
        
        The tables include 'fragments', 'fr_pairs', 'data_fr_pairs', 'data_frs', 'bonds', 'angles'.
        Parameters
        ----------
        data : bool, default True
            Print selected data of the tables if True. Otherwise, print only the column names of each table.
        """
        print('Array indicates the columns of the respective tables.')
        print('Afterwards selected information from the tables is printed.')
        print('\nTable fragments')
        info = self._cursor.execute('SELECT * FROM fragments').fetchall()
        print([description[0] for description in self._cursor.description])
        if data:
            print('ID    SMILES')
            for i in info:
                print(i[:2])

        print('\nTable Fragment Pairs')
        info = self._cursor.execute('SELECT * FROM fr_pairs').fetchall()
        print([description[0] for description in self._cursor.description])
        if data:
            print('ID_fr_pair    ID_fr1    ID_fr2    SMILES')
            for i in info:
                print(i[:4])

        print('\nTable Data Fragment Pairs.')
        info = self._cursor.execute('SELECT * FROM data_fr_pairs').fetchall()
        print([description[0] for description in self._cursor.description])
        if data:
            print('ID    datapoints')
            for i in info:
                print(i[0], deserialize(i[1])[0].shape[0])

        print('\nTable Data Fragments')
        info = self._cursor.execute('SELECT * FROM data_frs').fetchall()
        print([description[0] for description in self._cursor.description])
        if data:
            print('ID    datapoints')
            for i in info:
                print(i[0], deserialize(i[1])[0].shape[0])

        print('\nTable Bonds')
        info = self._cursor.execute('SELECT * FROM bonds').fetchall()
        print([description[0] for description in self._cursor.description])
        if data:
            print(info)

        print('\nTable Angles')
        info = self._cursor.execute('SELECT * FROM angles').fetchall()
        print([description[0] for description in self._cursor.description])
        if data:
            print(info)

    def isin_fr(self, fr):
        """
        Check if the provided fragment is stored in the database.

        If more than one fragment with the same SMILES representation is stored in the database, an error is raised.

        Parameters
        ----------
        fr : Fragment
            The fragment to check.

        Returns
        -------
        int or False
            The ID of the fragment if it is present in the database, False otherwise.
        """
        candidates = self._cursor.execute('SELECT fr_id FROM fragments WHERE smiles = ?',
                                          (fr.smiles,)).fetchall()
        if candidates:
            if len(candidates) > 1:
                logger = setup_logger(__name__)
                logger.error(f'Several fragments with the same SMILES representation are in the database.'
                             f'They have the ids: {candidates}.'
                             f'This should not be the case. Abort...')
                sys.exit(-1)
            return candidates[0][0]
        return False

    def _next_fr_id(self):
        """
        Get the next available fragment identifier (from the table 'fragments').

        Each fragment has a unique identifier. The identifiers are integers and start at 1.
        This function determines the largest identifier of the database and returns the next largest integer.
        If no fragment is stored in the database 1 is returned.

        Returns
        -------
        int
            The next available fragment identifier.
        """
        idx = self._cursor.execute('SELECT fr_id FROM fragments').fetchall()
        if not idx:
            return 1
        else:
            idx = np.array([i[0] for i in idx])
            return np.max(idx) + 1

    def get_fr(self, identifier, rng):
        """
        Retrieve the Fragment object from the database with the given identifier.

        Parameters
        ----------
        identifier : int
            The identifier of the fragment to retrieve.
        rng : np.random.default_rng()
            Default random number generator of numpy. Used for randomly selecting datapoints (see Notes section).

        Returns
        -------
        Fragment : The Fragment object with the given identifier if it exists in the database, otherwise False.

        Notes
        -----
            - This method retrieves a Fragment object from the 'fragments' table in the database.
            - If the fragment exists in the database, its attributes (atoms, A, internal_coords, smiles)
                are deserialized and used to create the Fragment object.
            - The 'models' attribute of the Fragment object is also set by deserializing the 'main_coords'
                and 'model' values from the database.
            - If the fragment has associated data in the 'data_frs' table, the 'zmatrix' and 'coords' attributes
                of the Fragment object are set by deserializing the corresponding values from the database.
            - If the number of datapoints in 'zmatrix' and 'coords' are greater than the 'n_datapoints' attribute
                of the Database object, a random subset of datapoints of size 'n_datapoints' is selected.
        """
        
        db_entry = self._cursor.execute('SELECT * FROM fragments WHERE fr_id = ?', (identifier,)).fetchone()
        if db_entry:
            smiles = db_entry[1]
            atoms, A, internal_coords = deserialize(*db_entry[2:5])
            fr = Fragment(atoms, A, internal_coords=internal_coords, smiles=smiles)
            main_coords, model = deserialize(*db_entry[5:])
            fr.set_models(main_coords, model)
        else:
            return False

        # Set data of fragment
        db_entry = self._cursor.execute('SELECT * FROM data_frs WHERE fr_id = ?', (identifier,)).fetchone()
        if db_entry is not None:
            zmatrix, coords = deserialize(*db_entry[1:])
            if len(zmatrix) > 0 and len(coords) > 0 and zmatrix.shape[0] > self.n_datapoints:
                choice = rng.choice(zmatrix.shape[0], size=self.n_datapoints, replace=False)
                zmatrix, coords = [
                    col_values[choice, :]
                    for col_values in [zmatrix, coords]
                ]
            if zmatrix is not None and coords is not None:
                fr.set_data(zmatrix, coords)
        return fr

    def get_fr_models(self, identifier):
        """
        Retrieve the models (main coordinates and their probabilities) of the fragment associated
        with the given identifier from the database (fragments table).

        Parameters
        ----------
        identifier : int
            The identifier of the fragment.

        Returns
        -------
        list or False
            Models (main coordinates and ML model) if the identifier exists and models are not None, False otherwise.
        """
        db_entry = self._cursor.execute('''SELECT main_coords, model FROM fragments WHERE fr_id = ?''',
                                        (identifier,)).fetchone()
        if db_entry is None:
            return False
        else:
            models = deserialize(*db_entry)

            for model in models:
                if model is None:
                    return False

            return models

    def add_fragment(self, fr):
        """
        Add a fragment to the database if it does not exist.

        Attributes of the fragments are serialized (pickled) before storing them in the database.

        Parameters
        ----------
        fr : Fragment
            The fragment to add to the database.

        Returns
        -------
        int
            Identifier of the fragment in the database.
        """
        if self.isin_fr(fr):
            logger = setup_logger(__name__)
            logger.warning("You wanted to add a fragment to the database. However, a fragment with the same SMILES "
                           "representation is already stored in the database. I will keep the original data.")
            return False

        fr_id = self._next_fr_id()
        smiles = fr.smiles
        atoms, A, internal_coords = serialize(fr.atoms, fr.A.values, fr.zmatrix)
        models = serialize(*fr.get_models())
        data = serialize(*fr.get_data())

        try:
            self._cursor.execute('INSERT INTO fragments VALUES (?, ?, ?, ?, ?, ?, ?)',
                                 (fr_id, smiles, atoms, A, internal_coords, *models))
            self._cursor.execute('INSERT INTO data_frs VALUES (?, ?, ?)', (fr_id, *data))
            self._connection.commit()
        except sqlite3.Error as err:
            logger = setup_logger(__name__)
            logger.error(f'Adding fragment to database failed due to error: {err}.')
            raise

        return fr_id

    def get_fr_ids(self, data_required=False):
        """
        Retrieve the fragment identifiers from the database.

        Parameters
        ----------
        data_required : bool, default False
            If True, only fragment identifiers containing simulation data will be returned.
            Otherwise, all identifiers are returned.

        Returns
        -------
        list or False
            List of fragment identifiers if found, False otherwise.
        """
        idx = self._cursor.execute('SELECT fr_id FROM fragments').fetchall()
        idx = [i[0] for i in idx]
        if data_required:
            idx_new = []
            for identifier in idx:
                db_entry = self._cursor.execute('SELECT * FROM data_frs WHERE fr_id = ?',
                                                (identifier,)).fetchone()
                if db_entry is not None:
                    zmatrix_data, coords_data = deserialize(*db_entry[1:])
                    if len(zmatrix_data) > 0 and len(coords_data) > 0:
                        idx_new.append(identifier)
            idx = idx_new
        if idx:
            return idx
        else:
            return False

    def isin_fr_pair(self, fr_pair):
        """
        Check if the provided fragment pair is stored in the database.

        The SMILES representation of the fragment pair and the individual fragments
        are compared to the database to account for different coarse-graining schemes.

        Parameters
        ----------
        fr_pair : FragmentPair
            The fragment pair to check.

        Returns
        -------
        tuple
            A tuple containing two values:
                1. The fragment pair identifier if it exists in the database, False otherwise.
                2. True if the fragments of the fragment pair in the database are reversed (see Notes section),
                   False otherwise.

        Notes
        -----
        The FragmentPair class has the attributes fr1 and fr2. They are unordered.
        Thus, the FragmentPair objects X and Y are identical if X.fr1 == Y.fr1 AND X.fr2 == Y.fr2 -> reverse is False
        Also, they are identical if X.fr1 == Y.fr2 AND X.fr2 == Y.fr1 -> reverse is True
        """
        candidates = self._cursor.execute('SELECT id, fr1_id, fr2_id FROM fr_pairs WHERE smiles = ?',
                                          (fr_pair.smiles,)).fetchall()
        if candidates:
            for id_, fr1_id, fr2_id in candidates:
                smiles_fr1 = self._cursor.execute('SELECT smiles FROM fragments WHERE fr_id = ?',
                                                  (fr1_id,)).fetchone()[0]
                smiles_fr2 = self._cursor.execute('SELECT smiles FROM fragments WHERE fr_id = ?',
                                                  (fr2_id,)).fetchone()[0]
                if fr_pair.fr1.smiles == smiles_fr1 and fr_pair.fr2.smiles == smiles_fr2:
                    return id_, False
                elif fr_pair.fr2.smiles == smiles_fr1 and fr_pair.fr1.smiles == smiles_fr2:
                    return id_, True
        return False, False

    def _next_fr_pair_id(self):
        """
        Get the next available fragment pair identifier (from the table 'fr_pairs').

        Each fragment pair has a unique identifier. The identifiers are integers and start at 1.
        This function determines the largest identifier of the database and returns the next largest integer.
        If no fragment pair is stored in the database 1 is returned.

        Returns
        -------
        int
            The next available fragment pair identifier.
        """
        idx = self._cursor.execute('SELECT id FROM fr_pairs').fetchall()
        if not idx:
            return 1
        else:
            idx = np.array([i[0] for i in idx])
            return np.max(idx) + 1

    def get_fr_pair(self, identifier, rng, reverse=False):
        """
        Retrieve the FragmentPair object from the database with the given identifier.

        Parameters
        ----------
        identifier : int
            The identifier of the fragment pair to retrieve.
        rng : np.random.default_rng()
            Default random number generator of numpy. Used for randomly selecting datapoints (see Notes section).
        reverse : bool
            True if the fragments should be reversed.

        Returns
        -------
        FragmentPair
            The FragmentPair object with the given identifier if it exists in the database, otherwise False.
        Notes
        -----
            - The FragmentPair class has the attributes fr1 and fr2. They are unordered.
              Thus, the FragmentPair objects X and Y are identical if X.fr1 == Y.fr1 AND X.fr2 == Y.fr2
              -> reverse is False
              Also, they are identical if X.fr1 == Y.fr2 AND X.fr2 == Y.fr1 -> reverse is True
            - If the fr_pair exists, the fragment IDs are used to retrieve the corresponding fragments from
              the 'fragments' table in the database. The fr_pair is then initialized with the retrieved fragments and
              the serialized connector information.
            - Fragment pair models are retrieved from the 'fr_pairs' table in the database
              and set in the fr_pair object.
            - Fragment pair data is retrieved from the 'data_fr_pairs' table in the database
              and set in the fr_pair object.
            - If the number of datapoints are greater than the 'n_datapoints' attribute of the Database object,
              a random subset of datapoints of size 'n_datapoints' is selected.
        """
        db_entry = self._cursor.execute('SELECT * FROM fr_pairs WHERE id = ?', (identifier,)).fetchone()
        if db_entry is not None:
            fr1 = self.get_fr(db_entry[1], rng)
            fr2 = self.get_fr(db_entry[2], rng)
            smiles = db_entry[3]

            # Initialize Connector
            atoms, elements, bond_types = deserialize(*db_entry[4:7])
            if reverse:
                atoms = np.flip(atoms)
                bond_types = np.flip(bond_types)
            con_four = Connector(atoms, bond_types)

            # Initialize fr_pair
            if reverse:
                fr_pair = FragmentPair(fr2, fr1, con_four, smiles)
                fr_pair.reverse = True
            else:
                fr_pair = FragmentPair(fr1, fr2, con_four, smiles)

            # Set models of fr_pair
            main_coords1, main_internal1, main_coords2, main_internal2, \
                main_dihedrals, model = deserialize(*db_entry[7:])
            if (main_coords1 is not None and main_internal1 is not None and main_coords2 is not None
                    and main_internal2 is not None and main_dihedrals is not None and model is not None):
                if reverse:
                    fr_pair.set_models(main_coords2, main_internal2,
                                          main_coords1, main_internal1, main_dihedrals, model)
                else:
                    fr_pair.set_models(main_coords1, main_internal1,
                                          main_coords2, main_internal2, main_dihedrals, model)
        else:
            return False

        # Set data of fr_pair
        db_entry = self._cursor.execute('SELECT * FROM data_fr_pairs WHERE id = ?', (identifier,)).fetchone()
        if db_entry is not None:
            zmatrix1, coords1, zmatrix2, coords2, X, dihedral = deserialize(*db_entry[1:])

            if zmatrix1.shape[0] > self.n_datapoints:  # sample only when population is greater than threshold
                # randomly sample 'n_datapoints' number of values from relevant columns in db_entry
                choice = rng.choice(zmatrix1.shape[0], size=self.n_datapoints, replace=False)
                zmatrix1, coords1, zmatrix2, coords2, X, dihedral = [
                    col_values[choice, :]
                    for col_values in [zmatrix1, coords1, zmatrix2, coords2, X, dihedral]
                ]

            if (zmatrix1 is not None and coords1 is not None and
                    zmatrix2 is not None and coords2 is not None and
                    X is not None and dihedral is not None):
                if reverse:
                    fr_pair.set_data(zmatrix2, coords2, zmatrix1, coords1, X, dihedral)
                else:
                    fr_pair.set_data(zmatrix1, coords1, zmatrix2, coords2, X, dihedral)
        return fr_pair

    def get_fr_pair_models(self, identifier, reverse=False):
        """
        Retrieve the models (main coordinates and internal coordinates of fragment 1 and 2,
        internal coordinates of the connector, and ML model) from the database (fr_pairs table).

        Parameters
        ----------
        identifier : int
            The identifier of the fragment pair.
        reverse : bool
            True if the models should be reversed.

        Returns
        -------
        list
            Fragment pair models.

        Notes
        -----
        The FragmentPair class has the attributes fr1 and fr2. They are unordered.
        Thus, the FragmentPair objects X and Y are identical if X.fr1 == Y.fr1 AND X.fr2 == Y.fr2 -> reverse is False
        Also, they are identical if X.fr1 == Y.fr2 AND X.fr2 == Y.fr1 -> reverse is True
        The order of the models has to be adjusted accordingly.
        """
        db_entry = self._cursor.execute('''SELECT main_coords1, main_internal, main_coords2, main_internal2, 
                                        main_dihedrals, model FROM fr_pairs WHERE id = ?''',
                                        (identifier,)).fetchone()
        if db_entry is None:
            return False
        else:
            models = deserialize(*db_entry)
            if reverse:
                idx = [2, 3, 0, 1, 4, 5]
                models = [models[i] for i in idx]
            for model in models:
                if model is None:
                    return False
            return models

    def add_fr_pair(self, fr_pair):
        """
        Add a fragment pair (and its corresponding fragments) to the database if it does not exist.

        Attributes of the fragment pair and its fragments are serialized (pickled) before storing them in the database.

        Parameters
        ----------
        fr_pair : FragmentPair
            The fragment pair to add to the database.

        Returns
        -------
        int
            Identifier of the fragment pair in the database.
        """
        if self.isin_fr_pair(fr_pair) != (False, False):
            logger = setup_logger(__name__)
            logger.warning("You wanted to add a fragment pair to the database. However, it already exists in the "
                           "database. I will keep the original data.")
            return False

        fr_pair_id = self._next_fr_pair_id()
        fr1_id = self.isin_fr(fr_pair.fr1)
        if not fr1_id:
            fr1_id = self.add_fragment(fr_pair.fr1)
        fr2_id = self.isin_fr(fr_pair.fr2)
        if not fr2_id:
            fr2_id = self.add_fragment(fr_pair.fr2)

        atoms, elements, bond_types = serialize(fr_pair.con.atoms, fr_pair.con.elements,
                                                fr_pair.con.bond_types)
        smiles = fr_pair.smiles
        models = serialize(*fr_pair.get_models())
        data = serialize(*fr_pair.get_data())

        try:
            self._cursor.execute('INSERT INTO fr_pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                 (fr_pair_id, fr1_id, fr2_id, smiles, atoms, elements, bond_types, *models))
            self._cursor.execute('INSERT INTO data_fr_pairs VALUES (?, ?, ?, ?, ?, ?, ?)', (fr_pair_id, *data))
            self._connection.commit()
        except sqlite3.Error as err:
            logger = setup_logger(__name__)
            logger.error(f'Adding data to table \'fr_pairs\' in database failed due to error: {err}.')
            raise
        return fr_pair_id

    def get_fr_pair_ids(self):
        """
        Retrieve the fragment pair identifiers from the database.

        Returns
        -------
        list or False
            List of fragment identifiers if found, False otherwise.
        """
        idx = self._cursor.execute('SELECT id FROM fr_pairs').fetchall()
        if idx:
            idx = [i[0] for i in idx]
            return idx
        else:
            return False

    def release(self, filename, delete=False):
        """
        Copy tables from the current database to a new database file.

        Parameters
        ----------
        filename : str
            The path of the new database file.
        delete : bool, default False
            If True, the original database file gets deleted after copying the tables.
        """
        if delete and os.path.isfile(filename):
            os.remove(filename)

        try:
            self._cursor.execute('ATTACH DATABASE ? AS new_db', (filename,))
            self.copy_table(self._cursor, 'fragments')
            self.copy_table(self._cursor, 'fr_pairs')
            self.copy_table(self._cursor, 'data_fr_pairs', delete=True)
            self.copy_table(self._cursor, 'data_frs', delete=True)
            self.copy_table(self._cursor, 'bonds')
            self.copy_table(self._cursor, 'angles')
            self._connection.commit()
        except sqlite3.Error:
            logger = setup_logger(__name__)
            logger.error('Could not copy database for release.')
            raise

    @staticmethod
    def copy_table(cursor, table, delete=False):
        """
        Copy a table from one database to another.

        Parameters
        ----------
        cursor : sqlite3.Cursor
            The cursor of the database where the table is copied from.
        table : str
            The name of the table to copy.
        delete : bool, default False
            If True, the original table get deleted after copying.
        """
        command = cursor.execute('''SELECT sql FROM sqlite_master
                                          WHERE type="table" AND name=?''', (table,)).fetchone()[0]
        command = re.sub(table, f'new_db.{table}', command)
        cursor.execute(command)
        if not delete:
            command = f'INSERT INTO new_db.{table} SELECT * FROM {table}'
            cursor.execute(command)

    def append_fr_pair_data(self, identifier, data_fr_pair, reverse=False):
        """
        Append fragment pair data to an existing entry in the database.

        First, the existing data from the database is retrieved using the given identifier.
        Then, the existing data is combined with the new data and the database entry is updated.
        The combination of the data is done by calling the 'combine_data' function.

        Parameters
        ----------
        identifier : int
            The identifier of the fragment pair in the database.
        data_fr_pair : list
            The fragment pair data to be appended.
        reverse : bool, default False
            True if the data should be reversed.

        Notes
        -----
        The FragmentPair class has the attributes fr1 and fr2. They are unordered.
        Thus, the FragmentPair objects X and Y are identical if X.fr1 == Y.fr1 AND X.fr2 == Y.fr2 -> reverse is False
        Also, they are identical if X.fr1 == Y.fr2 AND X.fr2 == Y.fr1 -> reverse is True
        The order of the data has to be adjusted accordingly.
        """
        if reverse:
            idx = [2, 3, 0, 1, 4, 5]
            data_fr_pair = [data_fr_pair[i] for i in idx]
        db_entry = self._cursor.execute('''SELECT zmatrix1, coords1, zmatrix2, coords2, X, dihedral 
                                    FROM data_fr_pairs WHERE id = ?''', (identifier,)).fetchone()
        data_new = combine_data(db_entry, data_fr_pair)

        self._cursor.execute('''UPDATE data_fr_pairs SET zmatrix1 = ?, coords1 = ?, zmatrix2 = ?, coords2 = ?, X = ?,
                             dihedral = ? WHERE id = ?''', (*data_new, identifier))
        self._connection.commit()

    def append_fr_data(self, identifier, data_fr):
        """
        Append fragment data to an existing entry in the database.

        First, the existing data from the database is retrieved using the given identifier.
        Then, the existing data is combined with the new data and the database entry is updated.
        The combination of the data is done by calling the 'combine_data' function.

        Parameters
        ----------
        identifier : int
            The identifier of the fragment pair in the database.
        data_fr : list
            The fragment data to be appended.
        """
        db_entry = self._cursor.execute('''SELECT zmatrix, coords FROM data_frs WHERE fr_id = ?''',
                                        (identifier,)).fetchone()
        data_new = combine_data(db_entry, data_fr)

        self._cursor.execute('''UPDATE data_frs SET zmatrix = ?, coords = ? WHERE fr_id = ?''',
                             (*data_new, identifier))
        self._connection.commit()

    def update_fr_pair_models(self, identifier, models, reverse=False):
        """
        Replace the models of a fragment pair in the database.

        Parameters
        ----------
        identifier : int
            The identifier of the fragment pair to be updated.
        models : tuple
            A tuple containing the new models.
        reverse : bool, default False
            If True, reverse the fragment ids before updating.

        Notes
        -----
        The FragmentPair class has the attributes fr1 and fr2. They are unordered.
        Thus, the FragmentPair objects X and Y are identical if X.fr1 == Y.fr1 AND X.fr2 == Y.fr2 -> reverse is False
        Also, they are identical if X.fr1 == Y.fr2 AND X.fr2 == Y.fr1 -> reverse is True
        The order of the fragments has to be adjusted accordingly.
        """
        if reverse:
            # Switch fragment ids. Necessary because model is dependent on the order
            id_fr1, id_fr2 = self._cursor.execute('SELECT fr1_id, fr2_id FROM fr_pairs WHERE id = ?',
                                                  (identifier,)).fetchone()
            self._cursor.execute('''UPDATE fr_pairs SET fr1_id = ?, fr2_id = ?''', (id_fr2, id_fr1))
        models_new = serialize(*models)
        self._cursor.execute('''UPDATE fr_pairs SET main_coords1 = ?, main_internal = ?, 
                            main_coords2 = ?, main_internal2 = ?, main_dihedrals = ?, 
                            model = ? WHERE id = ?''', (*models_new, identifier))
        self._connection.commit()

    def update_fr_models(self, identifier, models):
        """
        Replace the main_coords and model entries of a fragment record in the database.

        Parameters
        ----------
        identifier : int
            The identifier of the fragment pair to be updated.
        models : tuple
            A tuple containing the new models.
        """
        models_new = serialize(*models)
        self._cursor.execute('''UPDATE fragments SET main_coords = ?, model = ? WHERE fr_id = ?''',
                             (*models_new, identifier))
        self._connection.commit()

    def delete_fr_pair(self, identifier):
        """
        Delete a fragment pair and its corresponding simulation data from the database.

        Parameters
        ----------
        identifier : int
            The identifier of the fragment pair to be deleted.

        Returns
        -------
        bool
            True if the fragment pair was successfully deleted, False otherwise.
        """
        try:
            self._cursor.execute('DELETE FROM fr_pairs WHERE id = ?', (identifier,))
            self._cursor.execute('DELETE FROM data_fr_pairs WHERE id = ?', (identifier,))
            self.cleanup()
            self._connection.commit()
            return True
        except Exception as err:
            logger = setup_logger(__name__)
            logger.error(f'Deleting the fr_pair with id {identifier} from the database failed due to error: {err}.')
            return False

    def delete_fr(self, identifier):
        """
        Delete a fragment and its corresponding data from the database.

        Parameters
        ----------
        identifier : int
            The identifier of the fragment to be deleted.

        Returns
        -------
        bool
            True if the fragment was successfully deleted, False otherwise.
        """
        try:
            self._cursor.execute('DELETE FROM fragments WHERE fr_id = ?', (id_fr1,))
            self._cursor.execute('DELETE FROM data_frs WHERE fr_id = ?', (identifier,))
            self.cleanup()
            self._connection.commit()
            return True
        except Exception as err:
            logger = setup_logger(__name__)
            logger.error(f'Deleting the fragment with id {identifier} from the database failed due to error: {err}.')
            return False

    def cleanup(self):
        """
        Cleans up the database by checking for all fragment pairs and deleting any fragment pairs
        where the respective fragments no longer exist.
        """
        idx = self.get_fr_pair_ids()
        for identifier in idx:
            id_fr1, id_fr2 = self._cursor.execute('SELECT fr1_id, fr2_id FROM fr_pairs WHERE id = ?',
                                                  (identifier,)).fetchone()
            if not self.isin_fr(id_fr1) or not self.isin_fr(id_fr2):
                self.delete_fr_pair(identifier)

    def add_bond_data(self, bonds):
        """
        Add bond length data to the database.

        If the number of data points for a bond reaches 1000 in the database, new data is ignored.

        Parameters
        ----------
        bonds :dict
            A dictionary containing bond information. The keys are tuples representing the bond
            (element1, element2, bond_type), and the values are lists of bond lengths data points.
        """
        for key, val in bonds.items():
            # Sort key alphabetically
            if key[1] < key[0]:
                key = (key[1], key[0], key[2])

            data_points = len(val)
            entry_db = self._cursor.execute('SELECT * FROM bonds WHERE element1 = ? AND element2 = ? AND bond_type = ?',
                                            (*key,)).fetchone()
            if not entry_db:
                mean_val = sum(val) / data_points
                self._cursor.execute('INSERT INTO bonds VALUES (?, ?, ?, ?, ?)',
                                     (*key, mean_val, data_points))
            elif entry_db[4] < 1000:
                data_points_db = entry_db[4]
                mean_val_db = entry_db[3]
                mean_val = (sum(val) + mean_val_db * data_points_db) / (data_points + data_points_db)
                self._cursor.execute('''UPDATE bonds SET val = ?, datapoints = ?
                                     WHERE element1 = ? AND element2 = ? AND bond_type = ?''',
                                     (mean_val, data_points + data_points_db, *key))
            else:
                continue
            self._connection.commit()

    def get_bond_value(self, element1, element2, bond_type):
        """
        Retrieve the bond length of a specified bond.

        A warning is logged if the number of data points is less than 50.

        Parameters
        ----------
        element1 : str
            The first element.
        element2 : str
            The second element.
        bond_type : int
            The bond type. 1 for single, 2 for double, etc

        Returns
        -------
        float or None
            The bond length, None if no entry is found in the database.

        """
        if element1 < element2:
            entry_db = self._cursor.execute(
                'SELECT val, datapoints FROM bonds WHERE element1 = ? AND element2 = ? AND bond_type = ?',
                (element1, element2, int(bond_type))).fetchone()
        else:
            entry_db = self._cursor.execute(
                'SELECT val, datapoints FROM bonds WHERE element1 = ? AND element2 = ? AND bond_type = ?',
                (element2, element1, int(bond_type))).fetchone()

        if not entry_db:
            return None
        val, data_points = entry_db
        if data_points < 50:
            logger = setup_logger(__name__)
            logger.warning(f'There are only {data_points} data points for the bond {element1}-{element2}'
                           f'of type {bond_type}')
        return val

    def add_angle_data(self, angles):
        """
        Add angle data to the database.

        If the number of data points for a angle reaches 1000 in the database, new data is ignored.

        Parameters
        ----------
        angles :dict
            A dictionary containing angle information. The keys are tuples representing the angle
            (element1, element2, element3, bond_type1, bond_type2), and the values are lists of angle data points.
        """
        for key, val in angles.items():
            # Sort first and third entry of key alphabetically. Angle O-C-C is the same as C-C-O.
            if key[2] < key[0]:
                key = (key[2], key[1], key[0], key[4], key[3])

            data_points = len(val)
            entry_db = self._cursor.execute('''SELECT * FROM angles WHERE element1 = ? AND element2 = ? AND element3 = ?
                                            AND bond_type1 = ? AND bond_type2 = ?''',
                                            (*key,)).fetchone()
            if not entry_db:
                mean_val = sum(val) / data_points
                self._cursor.execute('INSERT INTO angles VALUES (?, ?, ?, ?, ?, ?, ?)',
                                     (*key, mean_val, data_points))
            elif entry_db[6] < 1000:
                data_points_db = entry_db[6]
                mean_val_db = entry_db[5]
                mean_val = (sum(val) + mean_val_db * data_points_db) / (data_points + data_points_db)
                self._cursor.execute('''UPDATE angles SET val = ?, datapoints = ? 
                                     WHERE element1 = ? AND element2 = ? AND element3 = ? 
                                     AND bond_type1 = ? AND bond_type2 = ?''',
                                     (mean_val, data_points + data_points_db, *key))
            else:
                continue
            self._connection.commit()

    def get_angle_value(self, element1, element2, element3, bond_type1, bond_type2):
        """
        Retrieve the value of a specified angle.

        A warning is logged if the number of data points is less than 50.

        Parameters
        ----------
        element1 : str
            The first element.
        element2 : str
            The second element.
        element3 : str
            The third element
        bond_type1 : int
            The bond type between element1 and element2. 1 for single, 2 for double, ...
        bond_type2 : int
            The bond type between element2 and element3. 1 for single, 2 for double, ...

        Returns
        -------
        float or None
            The angle value, None if no entry is found in the database.
        """
        if element1 < element3:
            entry_db = self._cursor.execute(
                '''SELECT val, datapoints FROM angles WHERE element1 = ? AND element2 = ? AND element3 = ? 
                AND bond_type1 = ? AND bond_type2 = ?''',
                (element1, element2, element3, int(bond_type1), int(bond_type2))).fetchone()
        else:
            entry_db = self._cursor.execute(
                '''SELECT val, datapoints FROM angles WHERE element1 = ? AND element2 = ? AND element3 = ? 
                AND bond_type1 = ? AND bond_type2 = ?''',
                (element3, element2, element1, int(bond_type2), int(bond_type1))).fetchone()
        if not entry_db:
            return None
        val, data_points = entry_db
        if data_points < 50:
            logger = setup_logger(__name__)
            logger.warning(f'There are only {data_points} data points for the angle {element1}-{element2}- {element3}'
                           f'of type {bond_type1} and {bond_type2}')
        return val
