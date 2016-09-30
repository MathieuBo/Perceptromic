import numpy as np
from multiprocessing import Pool
from module.save_multiproc import Database, BackUp
import pandas
import matplotlib.pyplot as plt
import time
from collections import OrderedDict


class Statistician(object):

    def __init__(self):
        self.database = Database('analysis_comb_avakas')

    def read_columns(self):
        return self.database.read_two_columns(column_names=['e_mean', 'v'])

    def analyse_combinations(self, threshold):

        mean_perf, var_names = self.read_columns()

        ordered_mean = np.argsort(mean_perf)

        ordered_var = list()

        for i in ordered_mean:
            ordered_var.append(var_names[i])

        select_comb = ordered_var[:int(len(mean_perf)*threshold)]

        return select_comb

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

        self.names = self.import_names(filename='names_105var')

        self.var_group = OrderedDict()
        self.var_group['behavior'] = {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 'color': 'dodgerblue'}

        self.var_group['dotblot'] = {'index': [16, 17, 18, 19, 20, 21, 22, 23], "color": 'sage'}

        self.var_group['synchrotron'] = {'index': [24, 25, 26, 27, 28, 29], 'color': 'darkorange'}

        self.var_group['histology'] = {'index': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                                                 62, 63, 64, 65, 66, 67, 77, 78, 79, 80, 81, 82, 83, 84,
                                                 85, 86, 87, 88, 89, 90, 91, 92, 93, 94], 'color': 'steelblue'}

        self.var_group['wb'] = {"index": [14, 15, 68, 69, 70, 71, 72, 73, 74, 75, 76, 95, 96, 97, 98, 98,
                                          99, 100, 101, 102, 103, 104], 'color': 'teal'}

        self.var_group['biomarkers'] = {'index': [30, 31, 32, 33], 'color': 'indianred'}

        self.var_group['qpcr'] = {'index': [34, 35], 'color': "gray"}

        self.total_list = np.arange(105)

        self.indexes_unusual = np.asarray(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 25, 26, 27, 28, 29, 36, 45,
             46, 47, 49, 50, 52, 53, 54, 57, 58, 59, 60, 61, 66, 77, 81, 82, 83, 85, 87, 88])

        self.indexes_usual = np.setdiff1d(self.total_list, self.indexes_unusual)

        self.indexes_nosyn = np.asarray(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27, 28, 29, 69, 70, 71, 72,
             73, 74, 75, 76, 98, 99, 100, 101, 102, 103, 104])

        self.indexes_syn = np.setdiff1d(self.total_list, self.indexes_nosyn)

        self.indexes = {'usual': self.indexes_usual, 'unusual': self.indexes_unusual,
                        'syn': self.indexes_syn, 'not_syn': self.indexes_nosyn}

    def get_color(self, selected_var):

        color = list()

        for i in selected_var:
            for j in self.var_group:
                if i in self.var_group[j]['index']:
                    color.append(self.var_group[j]['color'])

        return color

    def get_technics(self, best_var):

        output = OrderedDict()
        output['usual'] = {}
        output['unusual'] = {}
        output['syn'] = {}
        output['not_syn'] = {}

        for i in output:

            techs = OrderedDict()
            techs['behavior'] = 0
            techs['dotblot'] = 0
            techs['synchrotron'] = 0
            techs['histology'] = 0
            techs['wb'] = 0
            techs['biomarkers'] = 0
            techs['qpcr'] = 0

            for j in best_var:

                if j in self.indexes[i]:

                    for k in self.var_group:

                        if j in self.var_group[k]['index']:

                            techs[k] += 1

            total = sum(techs.values())

            for j in techs:
                techs[j] = 100 * techs[j] / total

            output[i] = techs

        names = output.keys()

        data_for_graph = np.empty((7, 4), dtype='float64')

        for var_type, i in enumerate(names):
            for tech, j in enumerate(output[i]):

                data_for_graph[tech, var_type] = output[i][j]

        return names, data_for_graph

    def find_best_var(self, threshold):

        selected_comb = self.analyse_combinations(threshold=threshold)

        best_var = []
        for i in selected_comb:
            val = i[1:-1].split(',')
            for j in val:
                best_var.append(int(j))

        return best_var

    def graph_freq(self, threshold):

        best_var = self.find_best_var(threshold=threshold)

        select_var, counts = np.unique(best_var, return_counts=True)

        print('Number of absent variable in the top 1% : {}'.format(len(np.setdiff1d(self.total_list, select_var))))

        counts_ord = counts.copy()
        y = np.sort(counts_ord)[::-1]

        y = y/(np.sum(y)/3)

        x = np.arange(len(select_var))

        labels = select_var[np.argsort(counts)[::-1]]

        color = self.get_color(labels)

        name_var = []
        for i in labels:
            name_var.append(self.names[i])

        # Plot
        plt.bar(x+.25, y, color=color)
        plt.xticks(x+.5, name_var, rotation='vertical')
        plt.xlim((0, 106))
        plt.tight_layout()
        plt.show()

    def graph_method(self, threshold):

        best_var = self.find_best_var(threshold=threshold)

        names, data_for_graph = self.get_technics(best_var)

        y_pos = np.arange(len(names))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        colors = []
        for i in self.var_group:
            colors.append(self.var_group[i]['color'])

        # Stacked histogram
        patch_handles = []
        left = np.zeros(len(names))

        for i, d in enumerate(data_for_graph):
            patch_handles.append(ax.barh(y_pos, d,
                                         color=colors[i], align='center',
                                         left=left))

            left += d

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Methodology usefullness')

        x_ticks = np.arange(110, step=10)
        ax.set_xticks(x_ticks)

        plt.show()


class Analyst:

    def __init__(self, threshold):

        self.threshold = threshold

        self.d = DataClassifier()

    def run_analysis(self):

        print("########################")
        print('Analysis')
        print(time.strftime("%d/%m/%Y"))
        print(time.strftime("%H:%M:%S"))
        print("########################\n")

        print("Analysis of frequence")
        self.d.graph_freq(threshold=self.threshold)

        print("\nAnalysis of methods")
        self.d.graph_method(threshold=self.threshold)


if __name__ == "__main__":

    analyst = Analyst(threshold=.01)

    analyst.run_analysis()
