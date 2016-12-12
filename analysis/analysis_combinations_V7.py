import numpy as np
from itertools import combinations
from os import path, mkdir
from tqdm import tqdm
from module.savev2 import Database
from sqlite3 import connect


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

        print("Begin importation of data.")
        values, selected_var = get_data_from_db(
            database_name=input_database,
            database_folder=database_folder)

        np.save(file=values_file, arr=values)
        np.save(file=selected_var_file, arr=selected_var)

    else:

        print("Load data from npy files.")
        values = np.load(file=values_file)
        selected_var = np.load(file=selected_var_file)

    return values, selected_var


# ----- Analysis function that organize stuff to do ------ #


def analyse(explanans_size, n_variable, n_network, output_database, input_database, temporary_files_folder,
            database_folder):

    comb_list = [str(i) for i in combinations(np.arange(explanans_size), n_variable)]

    values, selected_var = get_values_and_selected_variables(
        temporary_files_folder=temporary_files_folder, input_database=input_database,
        database_folder=database_folder)

    post_learning_dic = dict()

    print("Reformat data.")
    for i in comb_list:
        post_learning_dic[i] = []

    for idx, sel_var in enumerate(selected_var):

        post_learning_dic[sel_var].append(values[idx, 0])

    print()
    print("Prepare backup")
    backup = []
    idx = 0
    for key, value in tqdm(post_learning_dic.items()):

        mean = np.mean(post_learning_dic[key])
        sem = mean/np.sqrt(n_network)
        backup.append([idx, key, mean, sem])
        idx += 1

    print("Do backup.")

    connection = connect("{}/{}.db".format(database_folder, output_database))
    cursor = connection.cursor()

    q = "CREATE TABLE data (ID INTEGER PRIMARY KEY, " \
        "v TEXT, e_mean FLOAT, sem FLOAT)"

    cursor.execute(q)
    q = "INSERT INTO data VALUES (?, ?, ?, ?)"
    cursor.executemany(q, backup)
    connection.commit()
    connection.close()


# ----- Main ------ #


def main():

    explanans_size = 130
    n_variable = 3
    n_network = 50

    temporary_files_folder = path.expanduser("~/Desktop")
    database_folder = path.expanduser("~/Desktop")
    input_db = "combinations_061216"
    output_db = "analysis_comb_avakas_061216"

    analyse(
        explanans_size=explanans_size, n_variable=n_variable,
        n_network=n_network,
        input_database=input_db, output_database=output_db,
        database_folder=database_folder,
        temporary_files_folder=temporary_files_folder)


if __name__ == "__main__":

    main()
