import numpy as np
from itertools import combinations
from os import path, mkdir
from multiprocessing import Pool, Value, cpu_count
from time import time
from module.savev2 import BackUp, Database


# Global variables that will be used for parallel computation


shared_selected_variables = None
shared_values = None
counter = None
n = None
shared_beginning_time = None


# ----- Get data ------ #


def get_data_from_db(database_name, database_folder):

    db = Database(database_name=database_name, database_folder=database_folder)
    raw_data = db.read_columns(column_list=["post_learning_test", "index_test", "selected_var"])

    n_rows = len(raw_data)
    n_columns = len(raw_data[0])
    data_as_array = np.asarray(raw_data)
    data_as_array.reshape((n_rows, n_columns))

    values = np.zeros((n_rows, n_columns - 1))
    values[:] = data_as_array[:, :2]
    selected_var = data_as_array[:, 2]

    return values, selected_var


def get_values_and_selected_variables(input_database, database_folder, temporary_files_folder):

    values_file = "{}/{}_values.npy".format(temporary_files_folder, input_database)
    selected_var_file = "{}/{}_selected_variables.npy".format(temporary_files_folder, input_database)

    if not path.exists(values_file) or not path.exists(selected_var_file):

        if not path.exists(temporary_files_folder):
            mkdir(temporary_files_folder)

        beginning_time = time()
        print("Begin importation of data.")
        values, selected_var = get_data_from_db(
            database_name=input_database,
            database_folder=database_folder)

        np.save(file=values_file, arr=values)
        np.save(file=selected_var_file, arr=selected_var)

        end_time = time()

        print("Importation of data done.")
        print("Time needed for the importation of data: {}".format(convert_seconds_to_h_m_s(end_time-beginning_time)))

    else:

        print("Load data from npy files.")
        values = np.load(file=values_file)
        selected_var = np.load(file=selected_var_file)

    return values, selected_var


# ----- Analysis function that organize stuff to do ------ #


def analyse(explanans_size, n_variable, output_database, input_database, temporary_files_folder,
            database_folder, n_worker):

    comb_list = [i for i in combinations(np.arange(explanans_size), n_variable)]

    values, selected_var = get_values_and_selected_variables(
        temporary_files_folder=temporary_files_folder, input_database=input_database,
        database_folder=database_folder)

    # ---- #
    # Set global variables that will serve for each worker.
    global shared_values
    global shared_selected_variables
    global counter
    global n
    global shared_beginning_time

    shared_values = values

    shared_selected_variables = selected_var

    counter = Value("i", 0)

    n = len(comb_list)

    # ---- #
    # Launch a pool of worker for doing the computations.

    print("Begin computations.")

    shared_beginning_time = time()

    pool = Pool(processes=n_worker)
    results = pool.map(func=compute, iterable=comb_list)

    end_time = time()

    print()
    print("Time needed for computations: {}.".format(convert_seconds_to_h_m_s(end_time-shared_beginning_time)))

    # ---- #
    # Save computations in a database
    b = BackUp(output_database, "data", database_folder=database_folder)
    b.save(results)


# ----- Computation function that will be used in parallel ------ #


def compute(variable_set):

    # ---- #
    # Do the analysis

    dic = dict()

    boolean = shared_selected_variables[:] == '{}'.format(variable_set)
    val = shared_values[boolean]
    mean_post_learning_test = np.mean(val[:, 0])

    dic["shared_values"] = variable_set
    dic["e_mean"] = mean_post_learning_test
    dic['sem'] = mean_post_learning_test / np.sqrt(50)
    dic['index_mean'] = np.mean(val[:, 1])

    # ---- #
    # Do some print for having an idea how the things go

    with counter.get_lock():
        counter.value += 1
        k = counter.value

    actual_time = time()

    progress = np.round(k/n * 100, decimals=2)
    remaining_time = (actual_time - shared_beginning_time) * (1/k) * (n-k)
    txt = "Progress: {}% [estimated remaining_time: {}.]".format(
        progress, convert_seconds_to_h_m_s(remaining_time))

    # Sophisticated print for having all in one single line
    print("\r{}".format(txt), end=" ", flush=True)

    return dic


# ----- Useful functions ------ #

def convert_seconds_to_h_m_s(seconds):

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


# ----- Main ------ #


def main():

    explanans_size = 120
    n_variable = 3

    temporary_files_folder = path.expanduser("~/Desktop")
    database_folder = path.expanduser("~/Desktop")
    input_db = "combinations_0"
    output_db = 'analysis_comb_avakas_061216'

    analyse(
        explanans_size=explanans_size, n_variable=n_variable,
        input_database=input_db, output_database=output_db,
        database_folder=database_folder,
        n_worker=cpu_count(), temporary_files_folder=temporary_files_folder)


if __name__ == "__main__":

    main()
