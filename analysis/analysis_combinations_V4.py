import numpy as np
from itertools import combinations
from os import path, mkdir

from multiprocessing import Pool, Value
from module.savev2 import BackUp, Database
import pickle

from time import time


class Statistician(object):

    counter = Value('i', 0)

    def __init__(self, explanans_size, n_variable):

        self.values = None
        self.selected_var = None

        self.comb_list = self.get_variable_set(explanans_size=explanans_size, n_variable=n_variable)

        self.max_counter = len(self.comb_list)

        self.frequency = 300

    @staticmethod
    def convert_seconds_to_h_m_s(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    @staticmethod
    def get_variable_set(explanans_size, n_variable):
        return [i for i in combinations(np.arange(explanans_size), n_variable)]

    @staticmethod
    def get_data_from_db(database_name):

        db = Database(database_name=database_name)
        raw_data = db.read_columns(column_list=["post_learning_test", "index_test", "selected_var"])

        n_rows = len(raw_data)
        n_columns = len(raw_data[0])
        data_as_array = np.asarray(raw_data)
        data_as_array.reshape((n_rows, n_columns))

        values = np.zeros((n_rows, n_columns - 1))
        values[:] = data_as_array[:, :2]
        selected_var = data_as_array[:, 2]

        return values, selected_var

    def compute_data(self, variable_set):

        dic = dict()

        boolean = self.selected_var == '{}'.format(variable_set)
        val = self.values[boolean]
        mean_post_learning_test = np.mean(val[:, 0])

        dic["v"] = variable_set
        dic["e_mean"] = mean_post_learning_test
        dic['sem'] = mean_post_learning_test / np.sqrt(50)
        dic['index_mean'] = np.mean(val[:, 1])

        self.counter.value += 1

        if self.counter.value % self.frequency == 0:

            print("Progress: {}%".format(np.round(self.counter.value / self.max_counter * 100, decimals=2)))

        return dic

    def db_to_pic(self, input_database):

        pic_folder = '../../results/pick'

        if not path.exists(pic_folder):

            print("Creating directory for pickles")
            mkdir(pic_folder)

            beginning_time = time()
            print("BEGIN IMPORT")
            self.values, self.selected_var = self.get_data_from_db(database_name=input_database)

            np.save(file="{}/values.p".format(pic_folder), arr= self.values)
            np.save(file="{}/sel_var.p".format(pic_folder), arr=self.selected_var)

            intermediate_time = time()

            print("IMPORT FINISHED")
            print("time : {}".format(self.convert_seconds_to_h_m_s(intermediate_time-beginning_time)))

            print("Pickles saved")

    def analyse(self, output_database, input_database, n_worker):

        self.db_to_pic(input_database=input_database)

        dic_folder = "../../results/dic/"
        assert path.exists(dic_folder)

        # for variable_set in self.comb_list:
        #
        #     dic = dict()
        #
        #     str_variable_set = '{}'.format(variable_set)
        #     boolean = self.selected_var == str_variable_set
        #
        #     val = self.values[boolean]
        #     mean_post_learning_test = np.mean(val[:, 0])
        #
        #     dic["v"] = variable_set
        #     dic["e_mean"] = mean_post_learning_test
        #     dic['sem'] = mean_post_learning_test / np.sqrt(50)
        #     dic['index_mean'] = np.mean(val[:, 1])
        #
        #     with open("{}/dic_{}.p".format(dic_folder, str_variable_set), mode='wb') as file:
        #         pickle.dump(file=file, obj=dic)
        #
        # end_time = time()
        # print("time : {}".format(self.convert_seconds_to_h_m_s(end_time - intermediate_time)))

        #b = BackUp(output_database, "data")
        #b.save(results)


if __name__ == "__main__":

    s = Statistician(explanans_size=130, n_variable=3)

    s.analyse(input_database="combinations_061216", output_database='analysis_comb_avakas_061216',
              n_worker=6)