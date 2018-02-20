import numpy as np
from module.save_multiproc import Database
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
import pickle


class Statistician(object):

    def __init__(self, db_name):
        self.database = Database(db_name)

    def read_columns(self):
        return self.database.read_two_columns(column_names=['e_mean', 'v'])

    def analyse_combinations(self, threshold):

        mean_perf, var_names = self.read_columns()

        ordered_mean = np.argsort(mean_perf)

        ordered_var = list()
        ordered_perf = list()

        for i in ordered_mean:
            ordered_var.append(var_names[i])
            ordered_perf.append(mean_perf[i])

        select_comb = ordered_var[:int(len(ordered_mean)*threshold)]
        select_comb_perf = ordered_perf[:int(len(ordered_mean)*threshold)]

        return select_comb, select_comb_perf

    @staticmethod
    def import_names(filename, random):

        names = np.loadtxt('../../var_combination/{}.txt'.format(filename), dtype='str')

        name_list = list()

        for i in names:
            name_list.append(i[1:-1])

        name_list = name_list[:-3]

        for i in np.arange(random):
            name_list.append("random{}".format(i))

        return name_list


class DataClassifier(Statistician):

    def __init__(self, db_name, total_var, explanans, group_name):

        Statistician.__init__(self, db_name=db_name)

        self.group = group_name

        self.names = self.import_names(filename='names_081417', random=total_var-explanans)

        self.total_list = np.arange(total_var)

        self.random_col = np.arange(start=explanans, stop=total_var)

        # Method type and associated color per variable
        self.colors = pickle.load(open('../../var_combination/var_colors.p', 'rb'))

    def get_color(self, selected_var):

        color = list()

        for i in selected_var:
            var = self.names[i]
            if 'rand' in var:
                color.append('#000000')
            else:
                color.append(self.colors[var])

        return color

    def find_best_var(self, threshold, group_name):

        selected_comb, selected_comb_perf = self.analyse_combinations(threshold=threshold)

        # Save the list of combinations as txt file
        with open('../../var_combination/selected_comb_{}.txt'.format(group_name), 'w') as file:
            for i, j in zip(selected_comb, selected_comb_perf):
                file.write('{i}\t{j}\n'.format(i=i, j=j))

        best_var = []
        for i in selected_comb:
            val = i[1:-1].split(',')
            for j in val:
                best_var.append(int(j))

        return best_var

    def graph_freq(self, threshold):

        best_var = self.find_best_var(threshold=threshold, group_name=group_name)

        selected_var = np.unique(best_var)
        print('Number of absent variable in the top 1% : {}'.format(len(np.setdiff1d(self.total_list, selected_var))))

        bstp_results = OrderedDict()
        for i in selected_var:
            bstp_results[i] = []

        # Bootstrap on the list of best variables
        n_bootstrap = 100

        for _ in np.arange(n_bootstrap):

            bstp = np.random.choice(best_var, size=len(best_var), replace=True)
            bstp_var, bstp_counts = np.unique(bstp, return_counts=True)

            for j, k in enumerate(selected_var):
                bstp_results[k].append(bstp_counts[j])

        var_for_order = []
        means = []
        ci = []
        for i in bstp_results.keys():
            m = np.mean(bstp_results[i])
            var_for_order.append(i)
            means.append(m)
            ci.append(2.575*np.std(bstp_results[i])/np.sqrt(n_bootstrap))

        name_var = []
        ordered_var = []
        for i in np.argsort(means)[::-1]:
            name_var.append(self.names[i])
            ordered_var.append(i)

        colors = self.get_color(selected_var=ordered_var)

        ordered_means = []
        ordered_ci = []
        for i in ordered_var:
            ordered_means.append(means[i])
            ordered_ci.append(ci[i])

        # Plot with performance

        plt.figure(figsize=(10, 5))
        pos = 0
        for m, c, color in zip(ordered_means, ordered_ci, colors):
            plt.bar(pos, m, width=.75, yerr=c, color=color)
            pos += 1
        plt.xlim((-.5, len(ordered_means)+.5))
        plt.ylabel('Counts in the top 1%')
        plt.title('Ordered abundance in top 1% +/- 99% CI')
        plt.savefig('../../var_combination/results/full_plot_{}.png'.format(self.group))
        plt.savefig('../../var_combination/results/full_plot_{}.svg'.format(self.group))
        plt.show()

        # Plot without random variables

        names_wo_rand = []
        means_wo_rand = []
        ci_wo_rand = []
        colors_wo_rand = []

        for name, m, c, col in zip(name_var, ordered_means, ordered_ci, colors):
            if 'rand' not in name:
                names_wo_rand.append(name)
                means_wo_rand.append(m)
                ci_wo_rand.append(c)
                colors_wo_rand.append(col)

        # Save the first 20 variables and associated couns in a txt file
        with open("../../var_combination/nodes_{}.txt".format(self.group), 'w') as file:
            for i, j in zip(names_wo_rand[:20], means_wo_rand[:20]):
                file.write('{}\t{}\n'.format(i, j))

        fig, ax = plt.subplots(figsize=(20, 5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.bar(np.arange(len(names_wo_rand)), means_wo_rand, width=.75, yerr=ci_wo_rand, color=colors_wo_rand)
        plt.xticks(np.arange(len(names_wo_rand)), names_wo_rand, rotation='vertical')
        plt.xlim((-.5, 163.5))
        plt.ylabel('Counts in the top 1%')
        plt.title('Ordered abundance in top 1% +/- 99% CI')
        plt.tight_layout()
        plt.savefig('../../var_combination/results/plot_{}.png'.format(self.group))
        plt.savefig('../../var_combination/results/plot_{}.svg'.format(self.group))
        plt.show()

        # Save ordered list of var for RRHO test
        with open('../../var_combination/results/order_{}.txt'.format(self.group), 'w') as file:
            for i, j in zip(names_wo_rand, means_wo_rand):
                    file.write('{}\t{}\n'.format(i, j))

        # Plots with only 20 first variables

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.bar(np.arange(20), means_wo_rand[:20], width=.75, yerr=ci_wo_rand[:20], color=colors_wo_rand[:20])
        plt.xticks(np.arange(20), names_wo_rand[:20], rotation='vertical')
        plt.xlim((-.5, 19.5))
        plt.ylabel('Counts in the top 1%')
        plt.title('Ordered abundance in top 1% +/- 99% CI')
        plt.tight_layout()
        plt.savefig('../../var_combination/results/plot_{}_top20.png'.format(self.group))
        plt.savefig('../../var_combination/results/plot_{}_top20.svg'.format(self.group))
        plt.show()


class Analyst:

    def __init__(self, threshold, total_var, explanans, db_name, group_name):

        self.threshold = threshold

        self.d = DataClassifier(db_name=db_name, total_var=total_var, explanans=explanans, group_name=group_name)

    def run_analysis(self):

        print("########################")
        print('Analysis')
        print(time.strftime("%d/%m/%Y"))
        print(time.strftime("%H:%M:%S"))
        print("########################\n")

        print("Analysis of frequence")
        self.d.graph_freq(threshold=self.threshold)


if __name__ == "__main__":

    total_number_var = 326
    explanans = 163

    # Analysis for lb group
    # group_name = 'lb'
    # database = 'analysis_combinations_101617'

    # Analysis for nolb group
    group_name = 'nolb'
    database = 'analysis_combinations_nolb_111317'

    # Frequency analysis
    analyst = Analyst(db_name=database, threshold=.01, total_var=total_number_var, explanans=explanans, group_name=group_name)
    analyst.run_analysis()

