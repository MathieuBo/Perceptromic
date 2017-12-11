import numpy as np
from itertools import combinations
from os import path, mkdir, listdir
from tqdm import tqdm
from sqlite3 import connect


class Statistician(object):

    def __init__(self, database_folder, temporary_files_folder,
                 input_database, output_database, explanans_size, n_variable, n_network):

        # --- Files and folders
        self.database_folder = database_folder
        self.temporary_files_folder = temporary_files_folder
        self.input_database = input_database
        self.output_database = output_database

        # --- Parameters
        self.temporary_files_folder = temporary_files_folder
        self.explanans_size = explanans_size
        self.n_variable = n_variable
        self.n_network = n_network

        # --- Data containers
        self.nb_combinations = len([i for i in combinations(np.arange(explanans_size*2), 3)])
        self.backup = []

        # --- Data identifiers
        self.idx = 0

    # ----- Get data ------ #

    def get_data_from_db(self):

        db_path = "{}".format(self.database_folder)
        assert path.exists(db_path), "Wrong path for the input database. Please correct it."

        list_db_name = [i for i in listdir(db_path) if i[-3:] == ".db"]

        # Create arrays with the first db
        first_db = '{}/{}'.format(db_path, list_db_name[0])
        connection = connect(first_db)
        cursor = connection.cursor()

        q = "SELECT post_learning_test, selected_var from data"
        cursor.execute(q)
        content = np.asarray(cursor.fetchall())
        connection.close()

        # Organize data and store them in a list
        db_init = self.reformat_data(data=content)
        self.prepare_backup(data=db_init)

        print('First DB read - Array created')

        # Append data for the other database by concatenating arrays
        for db_name in tqdm(list_db_name[1:]):
            next_db_path = '{}/{}'.format(db_path, db_name)
            connection = connect(next_db_path)
            cursor = connection.cursor()

            q = "SELECT post_learning_test, selected_var from data"
            cursor.execute(q)
            content = np.asarray(cursor.fetchall())
            connection.close()

            db_follow = self.reformat_data(data=content)
            self.prepare_backup(data=db_follow)

    # ----- Analysis properly speaking ------ #

    def run(self):

        assert not path.exists("{}/{}.db".format(self.database_folder, self.output_database)), \
            "Output database already exists. Please erase it or give it a new name."

        print("Reformat data and prepare backup")
        self.get_data_from_db()

        print()
        print("Do backup.")
        self.do_backup()

    @staticmethod
    def reformat_data(data):

        organized_data = dict()
        comb_list = np.unique(data[:, 1])

        for i in comb_list:
            organized_data[i] = []

        for sel_var, post_learning_res in zip(data[:, 1], data[:, 0]):
            organized_data[sel_var].append(float(post_learning_res))

        return organized_data

    def prepare_backup(self, data):

        for key in data.keys():

            try:
                mean = np.mean(data[key])
            except Exception:
                print(data[key])
                raise Exception
            sem = mean / np.sqrt(self.n_network)
            self.backup.append([self.idx, key, mean, sem])
            self.idx += 1

    def do_backup(self):

        # Establish connection with output database
        connection = connect("{}/{}.db".format(self.database_folder, self.output_database))
        cursor = connection.cursor()

        # Create table
        q = "CREATE TABLE data (ID INTEGER PRIMARY KEY, " \
            "v TEXT, e_mean FLOAT, sem FLOAT)"
        cursor.execute(q)

        # Fill table
        q = "INSERT INTO data VALUES (?, ?, ?, ?)"
        cursor.executemany(q, self.backup)

        # Close connection with output database
        connection.commit()
        connection.close()


# ----- Useful functions ------ #


def convert_seconds_to_h_m_s(seconds):

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


# ----- Main ------ #


def main():

    explanans_size = 163
    randomness_size = explanans_size

    columns_size = explanans_size + randomness_size

    n_variable = 3
    n_network = 50

    temporary_files_folder = path.expanduser("~/Desktop")
    database_folder = "{}/results/".format(temporary_files_folder)
    input_db = "combinations_nolb_111317"
    output_db = "analysis_{}".format(input_db)

    s = Statistician(
        explanans_size=columns_size, n_variable=n_variable,
        n_network=n_network,
        input_database=input_db, output_database=output_db,
        database_folder=database_folder,
        temporary_files_folder=temporary_files_folder)
    s.run()


if __name__ == "__main__":

    main()
