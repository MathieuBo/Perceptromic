import numpy as np
from itertools import combinations
from multiprocessing import Pool
from module.save_multiproc import Database, BackUp
import pandas
import matplotlib.pyplot as plt
import time


class Statistician(object):

    def __init__(self):
        self.database = Database('analysis_comb_avakas')

    def compute(self, v):
        return self.database.read_column(column_name='e_mean', v=v)[0]

    @staticmethod
    def combinations_of_selected_indexes(indexes, size):

        return [i for i in combinations(indexes, size)]

    def analyse_combinations(self, pool, indexes):

        comb_var = self.combinations_of_selected_indexes(indexes=indexes, size=3)

        result = np.zeros(len(comb_var))
        result[:] = pool.map(self.compute, comb_var)

        return comb_var, result

    @staticmethod
    def import_names(filename):

        names = np.loadtxt('../../var_combination/{}.txt'.format(filename), dtype='str')

        name_list = list()

        for i in names:
            name_list.append(i[3:-2])

        return name_list


class DataClassifier(Statistician):

    def __init__(self):

        Statistician.__init__(self)

        self.best = None
        self.worst = None

        self.test_dic = None

        self.names = self.import_names(filename='names_105var')

        self.var_group = {
            'behavior': {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 'color': 'dodgerblue'},
            'dotblot': {'index': [16, 17, 18, 19, 20, 21, 22, 23], "color": 'sage'},
            'synchrotron': {'index': [24, 25, 26, 27, 28, 29], 'color': 'darkorange'},
            'histology': {'index': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                                    62, 63, 64, 65, 66, 67, 77, 78, 79, 80, 81, 82, 83, 84,
                                    85, 86, 87, 88, 89, 90, 91, 92, 93, 94], 'color': 'steelblue'},
            'wb': {"index": [14, 15, 68, 69, 70, 71, 72, 73, 74, 75, 76, 95, 96, 97, 98, 98,
                             99, 100, 101, 102, 103, 104], 'color': 'teal'},
            'biomarkers': {'index': [30, 31, 32, 33], 'color': 'indianred'},
            'qpcr': {'index': [34, 35], 'color': "gray"}
        }

    def classification(self, indexes, table_name, pool):

        comb_var, means = self.analyse_combinations(pool, indexes=indexes)

        best_mean = min(means)
        worst_mean = max(means)

        self.test_dic = dict()
        results = []

        for i, j in zip(comb_var, means):

            dic = dict()

            dic["var_combination"] = i

            dic["e_mean"] = float(j)

            self.test_dic[i] = j

            results.append(dic)

            if j == best_mean:
                self.best = i
            if j == worst_mean:
                self.worst = i

        b = BackUp(database_name="classif_combinations", table_name=table_name)
        b.save(results)

    def get_best_worst(self, table_name):
        print('\n*************************')
        print('{}'.format(table_name))
        print('*************************')
        print('The best combination of {} is : {} '.format(table_name, self.best))
        for i in self.best:
            print(self.names[i])
        print('---------------------------')
        print('The worst combination of {} is : {} '.format(table_name, self.worst))
        for i in self.worst:
            print(self.names[i])

        print('---------------------------')
        print('Number of appearance of variables in the fifty best combinations')

        sorted_list = (sorted(self.test_dic, key=self.test_dic.__getitem__, reverse=False))
        print('Total : {} combinations \n'.format(len(sorted_list)))

        # selection = int(len(sorted_list)/100)  # Selection of the 1% best
        selection = 50

        print('Total considered combinations : {}'.format(selection))

        best_comb = sorted_list[:selection]

        best_values =[]
        for i in best_comb:
            best_values.append(self.test_dic[i])

        best_mean = np.mean(best_values)
        best_std = np.std(best_values)

        print("overvall performance of best combinations : {} +/- {}\n".format(best_mean, best_std))

        best_var = []
        for i in best_comb:
            for j in i:
                best_var.append(j)

        classif_best = pandas.value_counts(best_var)

        selected_indexes = np.unique(best_var)
        pie_data = []
        labels = []
        colors = []

        for i in selected_indexes:
            print('{} : {} '.format(self.names[i], classif_best[i]))
            labels.append(self.names[i])
            pie_data.append(classif_best[i])
            for key in self.var_group:
                if i in self.var_group[key]['index']:
                    colors.append(self.var_group[key]['color'])

        explode = np.zeros(len(pie_data))
        explode += 0.15

        plt.pie(pie_data, explode=explode, labels=labels,  colors=colors)
        plt.axis('equal')
        plt.savefig('../../graph_var_comb/{}.eps'.format(table_name), dpi=300)
        # plt.show()

        print('\n---------------------------')
        print('Number of appearance of variables in the fifty worst combinations')

        sorted_list = (sorted(self.test_dic, key=self.test_dic.__getitem__, reverse=False))
        print('Total : {} combinations \n'.format(len(sorted_list)))
        print('Total considered combinations : {}'.format(selection))

        worst_comb = sorted_list[-selection:]

        worst_values =[]
        for i in worst_comb:
            worst_values.append(self.test_dic[i])

        worst_mean = np.mean(worst_values)
        worst_std = np.std(worst_values)

        print("overvall performance of worst combinations : {} +/- {}\n".format(worst_mean, worst_std))

        worst_var = []
        for i in worst_comb:
            for j in i:
                worst_var.append(j)

        classif_worst = pandas.value_counts(worst_var)

        selected_indexes = np.unique(worst_var)

        for i in selected_indexes:
            print('{} : {} '.format(self.names[i], classif_worst[i]))

        print('*************************')


class SimpleAnalyse(Statistician):

    def __init__(self, n_var, n_workers=6):

        Statistician.__init__(self)

        self.pool = Pool(processes=n_workers)

        self.n_var = n_var

    def overall_mean(self, indexes=None):

        if indexes is None:
            indexes = np.arange(self.n_var)


        print('**********************')
        print('Usual var index are : {}'.format(indexes))

        comb_var, results = self.analyse_combinations(self.pool, indexes)

        print('Number of combinations used : {}'.format(len(comb_var)))

        mean = np.mean(results)
        sem = np.std(results)/np.sqrt(len(comb_var))

        print('Overall mean = {} +/- {}'.format(mean, sem))
        print('**********************')


class ClassificationAnalyst:

    def __init__(self, total_number_var, n_workers=6):

        self.d = DataClassifier()
        self.pool = Pool(processes=n_workers)

        self.total_list = np.arange(total_number_var)

        self.indexes_unusual = np.asarray(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 25, 26, 27, 28, 29, 36, 45,
             46, 47, 49, 50, 52, 53, 54, 57, 58, 59, 60, 61, 66, 77, 81, 82, 83, 85, 87, 88])

        self.indexes_usual = np.setdiff1d(self.total_list, self.indexes_unusual)

        self.indexes_nosyn = np.asarray(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27, 28, 29, 69, 70, 71, 72,
             73, 74, 75, 76, 98, 99, 100, 101, 102, 103, 104])

        self.indexes_syn = np.setdiff1d(self.total_list, self.indexes_nosyn)

    def run_analysis(self):

        print(time.strftime("%d/%m/%Y"))
        print(time.strftime("%H:%M:%S"))

        self.d.classification(indexes=self.indexes_usual, table_name='usual', pool=self.pool)
        self.d.get_best_worst(table_name='usual')

        self.d.classification(indexes=self.indexes_unusual, table_name='unusual', pool=self.pool)
        self.d.get_best_worst(table_name='unusual')

        self.d.classification(indexes=self.indexes_syn, table_name='Syn_related', pool=self.pool)
        self. d.get_best_worst(table_name='Syn_related')

        self.d.classification(indexes=self.indexes_nosyn, table_name='Not_syn_related', pool=self.pool)
        self.d.get_best_worst(table_name='Not_syn_related')


if __name__ == "__main__":

    # Create list with wanted indexes
    # s = SimpleAnalyse(n_workers=6, n_var=105)
    # s.overall_mean()

    c = ClassificationAnalyst(total_number_var=105, n_workers=6)
    c.run_analysis()



