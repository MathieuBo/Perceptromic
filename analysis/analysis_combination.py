import numpy as np
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool
from module.save import Database, BackUp
from sqlite3 import connect


class Statistician(object):

    def __init__(self, database_name):

        self.database = Database(database_name=database_name)

    def compute_mean(self, args):

        variable_set = args

        data = self.database.read_column(table_name="data", column_name="post_learning_test", selected_var=str(variable_set))
        return np.mean(data)

    def compute_sem(self, args):

        variable_set = args

        data = self.database.read_column(table_name="data", column_name="post_learning_test", selected_var=str(variable_set))
        return np.std(data)/np.sqrt(50)

    def compute_mean_index(self, args):

        variable_set = args

        data = self.database.read_column(table_name="data", column_name="index_test", selected_var=str(variable_set))
        return np.mean(data)

    def get_data_from_db(self):

        con = connect("results/combinations-copie.db")
        cursor = con.cursor()
        query = "SELECT `post_learning_test`, `index_test`, `selected_var` FROM data "
        raw_data = cursor.execute(query).fetchall()
        n_rows = len(raw_data)
        n_columns = len(raw_data[0])
        data_as_array = np.asarray(raw_data)
        data_as_array.reshape((n_rows, n_columns))

        values = np.zeros((n_rows, n_columns - 1))
        values[:] = data_as_array[:, :2]

        selected_var = data_as_array[:, 2]

        return values, selected_var

    def analyse(self, variable_sets, file_name, n_variable):

        # pool = Pool(processes=8)
        # means = pool.map(self.compute_mean, variable_sets)
        # sem = pool.map(self.compute_sem, variable_sets)
        # index = pool.map(self.compute_mean_index, variable_sets)

        # means = []
        # sem = []
        # index = []
        # for i in tqdm(range(len(variable_sets))):
        #     means.append(self.compute_mean(variable_sets[i]))
        #     sem.append(self.compute_sem(variable_sets[i]))
        #     index.append(self.compute_mean_index(variable_sets[i]))



        results = []
        for i, j, k, l in zip(means, variable_sets, sem, index):

            dic = dict()

            dic["v"] = j

            dic["e_mean"] = i

            dic['sem'] = k

            dic['index_mean'] = l

            dic['n_var'] = n_variable

            results.append(dic)

        b = BackUp(file_name, "data")
        b.save(results)


def get_variable_set(explanans_size, n_variable):

    return [i for i in combinations(np.arange(explanans_size), n_variable)]


def analyse_par_combinaison(explanans_size, nombre_variable, file_name):

    variable_sets = get_variable_set(explanans_size=explanans_size, n_variable=nombre_variable)
    s = Statistician(database_name='combinations-copie')
    s.analyse(variable_sets=variable_sets, file_name=file_name, n_variable=nombre_variable)

if __name__ == "__main__":

    # # Analyse de toutes les combinaisons
    # for i in [1, 2, 3]:
    #
    #     variable_sets = get_variable_set(explanans_size=52, n_variable=i)
    #
    #     s = Statistician(database_name='results_1continuous')
    #     s.analyse(variable_sets=variable_sets, file_name='analysis_comb_1continuous', n_variable= i)
    #

    # Analyse par combinaison
    # analyse_par_combinaison(explanans_size=52, nombre_variable=3, file_name='analysis_comb_3_tet')
    s = Statistician(database_name="combintations-copie")
    s.get_data_from_db()
