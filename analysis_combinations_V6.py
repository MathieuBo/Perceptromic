import numpy as np
from itertools import combinations, repeat

from os import path, mkdir

from multiprocessing import Pool, Value, Array, Manager, cpu_count
from module.savev2 import BackUp, Database

from time import time
import ctypes


sv = None
v = None
counter = None
max_counter = None


class Statistician(object):

    counter = Value('i', 0)

    def __init__(self, explanans_size, n_variable):

        self.comb_list = self.get_variable_set(explanans_size=explanans_size, n_variable=n_variable)

        self.selected_var = None
        self.values = None

    @staticmethod
    def get_variable_set(explanans_size, n_variable):
        return [i for i in combinations(np.arange(explanans_size), n_variable)]

    @staticmethod
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

    # def compute_data(self, variable_set):
    #
    #     dic = dict()
    #
    #     boolean = self.selected_var == '{}'.format(variable_set)
    #     val = self.values[boolean]
    #     mean_post_learning_test = np.mean(val[:, 0])
    #
    #     dic["v"] = variable_set
    #     dic["e_mean"] = mean_post_learning_test
    #     dic['sem'] = mean_post_learning_test / np.sqrt(50)
    #     dic['index_mean'] = np.mean(val[:, 1])
    #
    #     self.counter.value += 1
    #
    #     if self.counter.value % self.frequency == 0:
    #
    #         print("Progress: {}%".format(np.round(self.counter.value / self.max_counter * 100, decimals=2)))
    #
    #     return dic

    def get_values_and_selected_variables(self, input_database, database_folder, temporary_files_folder):

        values_file = "{}/{}_values.npy".format(temporary_files_folder, input_database)
        selected_var_file = "{}/{}_selected_variables.npy".format(temporary_files_folder, input_database)

        if not path.exists(values_file) or not path.exists(selected_var_file):

            if not path.exists(temporary_files_folder):
                mkdir(temporary_files_folder)

            beginning_time = time()
            print("BEGIN IMPORT")
            values, selected_var = self.get_data_from_db(
                database_name=input_database,
                database_folder=database_folder)

            np.save(file=values_file, arr=values)
            np.save(file=selected_var_file, arr=selected_var)

            end_time = time()

            print("IMPORT FINISHED")
            print("Time: {}".format(convert_seconds_to_h_m_s(end_time-beginning_time)))

        else:
            
            print("LOAD FROM TEMPORARY FILES")
            values = np.load(file=values_file)
            selected_var = np.load(file=sel_var_file)
        
        return values, selected_var

    def analyse(self, output_database, input_database, temporary_files_folder, database_folder, n_worker):

        values, selected_var = self.get_values_and_selected_variables(
            temporary_files_folder=temporary_files_folder, input_database=input_database,
            database_folder=database_folder)

        # m = Manager()
        self.selected_var = Array(ctypes.c_char_p, selected_var.size)
        # self.selected_var[:] = [str(i).encode() for i in selected_var]
        # self.selected_var.value = selected_var

        self.values = shared_zeros(values.shape[0], values.shape[1])
        self.values[:] = values

        # self.init_process(self.values, self.selected_var)

        global v
        v = self.values
        global sv
        sv = selected_var

        global counter
        counter = Value("i", 0)

        global max_counter
        max_counter = len(self.comb_list)

        # self.selected_var = Array()

        print("Begin computation")
        beginning_time = time()

        pool = Pool(processes=n_worker)
        results = pool.map(func=self.compute, iterable=self.comb_list)

        end_time = time()

        print("Time needed: {}.".format(convert_seconds_to_h_m_s(end_time-beginning_time)))

        print("Saving")

        b = BackUp(output_database, "data", database_folder=database_folder)
        b.save(results)

    @staticmethod
    def compute(variable_set):

        dic = dict()

        #  sel_var = np.asarray([i.decode() for i in sv[:]])

        boolean = sv[:] == '{}'.format(variable_set)
        val = v[boolean]
        mean_post_learning_test = np.mean(val[:, 0])

        dic["v"] = variable_set
        dic["e_mean"] = mean_post_learning_test
        dic['sem'] = mean_post_learning_test / np.sqrt(50)
        dic['index_mean'] = np.mean(val[:, 1])

        counter.value += 1
        txt = "Progress: {}%".format(np.round(counter.value / max_counter * 100, decimals=2))
        print("\r{}".format(txt), end=" ", flush=True)

        return dic


def convert_seconds_to_h_m_s(seconds):

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def shared_zeros(n1, n2):
    # create a  2D numpy array which can be then changed in different threads
    shared_array_base = Array(ctypes.c_double, n1 * n2)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(n1, n2)
    return shared_array


def main():

    temporary_files_folder = path.expanduser("~/Desktop")
    database_folder = path.expanduser("~/Desktop")
    input_db = "combinations_061216"
    output_db = 'analysis_comb_avakas_061216'

    s = Statistician(explanans_size=130, n_variable=3)

    s.analyse(input_database=input_db, output_database=output_db,
              database_folder=database_folder,
              n_worker=cpu_count(), temporary_files_folder=temporary_files_folder)


if __name__ == "__main__":

    main()
