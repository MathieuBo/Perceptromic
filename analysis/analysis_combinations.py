import numpy as np
from itertools import combinations
from os import path, mkdir
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
        self.post_learning_test = None  # Will be a numpy array with float
        self.selected_variables = None  # Will be a numpy array with strings
        self.formatted_data = dict()
        self.backup = []

    # ----- Get data ------ #

    def get_data_from_db(self):

        db_path = "{}/{}.db".format(self.database_folder, self.input_database)
        assert path.exists(db_path), "Wrong path for the input database. Please correct it."

        connection = connect(db_path)
        cursor = connection.cursor()

        q = "SELECT post_learning_test, selected_var from data"
        cursor.execute(q)
        content = np.asarray(cursor.fetchall())
        connection.close()

        self.post_learning_test = np.array(content[:, 0], dtype=float)
        self.selected_variables = content[:, 1]

    def get_post_learning_test_and_selected_variables(self, force):

        values_file = "{}/{}_post_learning_test.npy".format(self.temporary_files_folder, self.input_database)
        selected_var_file = "{}/{}_selected_variables.npy".format(self.temporary_files_folder, self.input_database)

        if not path.exists(values_file) or not path.exists(selected_var_file) or force:

            if not path.exists(self.temporary_files_folder):
                mkdir(self.temporary_files_folder)

            print("Import data from database.")
            self.get_data_from_db()

            np.save(file=values_file, arr=self.post_learning_test)
            np.save(file=selected_var_file, arr=self.selected_variables)

        else:

            print("Load data from npy files.")
            self.post_learning_test = np.load(file=values_file)
            self.selected_variables = np.load(file=selected_var_file)

    # ----- Analysis properly speaking ------ #

    def run(self):

        assert not path.exists("{}/{}.db".format(self.database_folder, self.output_database)), \
            "Output database already exists. Please erase it or give it a new name."

        # Can turn force argument to True if you want reload data from dab
        self.get_post_learning_test_and_selected_variables(force=True)

        print()
        print("Reformat data.")
        self.reformat_data()

        print()
        print("Prepare backup.")
        self.prepare_backup()

        print()
        print("Do backup.")
        self.do_backup()

    def reformat_data(self):

        comb_list = [str(i) for i in combinations(np.arange(self.explanans_size), self.n_variable)]

        for i in comb_list:
            self.formatted_data[i] = []

        for sel_var, post_learning_res in zip(self.selected_variables, self.post_learning_test):
            self.formatted_data[sel_var].append(post_learning_res)

        # --- Free memory
        self.selected_variables = None
        self.post_learning_test = None

    def prepare_backup(self):

        idx = 0
        for key, value in tqdm(self.formatted_data.items()):

            try:
                mean = np.mean(self.formatted_data[key])
            except Exception:
                print(self.formatted_data[key])
                raise Exception
            sem = mean / np.sqrt(self.n_network)
            self.backup.append([idx, key, mean, sem])
            idx += 1

        # --- Free memory
        self.formatted_data = None

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

    explanans_size = 130
    n_variable = 3
    n_network = 50

    temporary_files_folder = path.expanduser("~/Desktop")
    database_folder = path.expanduser("~/Desktop")
    input_db = "combinations_0"
    output_db = "analysis_{}".format(input_db)

    s = Statistician(
        explanans_size=explanans_size, n_variable=n_variable,
        n_network=n_network,
        input_database=input_db, output_database=output_db,
        database_folder=database_folder,
        temporary_files_folder=temporary_files_folder)
    s.run()


if __name__ == "__main__":

    main()
