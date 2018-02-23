import numpy as np
from itertools import combinations
from tqdm import tqdm
import os


class FileProcessor:

    def __init__(self, group_name):

        with open('../../var_combination/selected_comb_{}.txt'.format(group_name), 'r') as file:
            self.perf = file.read()

    def read(self):
        return self.perf.split('\n')[:-1]

    def read_only_comb(self):

        whole_file = self.perf.split('\n')[:-1]

        comb = list()

        for line in whole_file:
            to_add = line.split('\t')[0]
            comb.append(to_add)

        return comb

    @staticmethod
    def read_names():

        with open('../../var_combination/names_081417.txt', 'r') as file:
            names = file.read()

        names = names.split()

        id_names = {}

        for i, j in enumerate(names[:-3]):
            id_names[j[1:-1]] = i

        return names[:-3], id_names


class Frequencies:

    def __init__(self, group_name, explanans):

        self.fp = FileProcessor(group_name=group_name)

        self.explanans = explanans

        self.group = group_name

    def edit_file(self):

        with open('../../var_combination/results/order_{}.txt'.format(self.group), 'r') as file:
            freq = file.read()

        freq = freq.split('\n')[:-1]

        _, id_names = self.fp.read_names()

        freq_dict = {}

        for i in freq:
            i = i.split('\t')
            freq_dict['{}'.format(i[0])] = float(i[1])

        with open('../../var_combination/updated_nodes_{}.txt'.format(self.group), 'w') as file:
            file.write('id\tnames\tfreq\n')
            for var in freq_dict.keys():
                file.write('{id}\t{name}\t{val}\n'.format(id=id_names[var], name=var, val=freq_dict[var]))

        print('Update node list done!')


class Association:

    def __init__(self, group_name, explanans):

        self.fp = FileProcessor(group_name=group_name)

        self.explanans = explanans

        self.group_name = group_name

        self.list_duo = [i for i in combinations(np.arange(self.explanans), 2)]

    def performance_duo(self):

        perf = self.fp.read()

        comb = []
        perf_val = []

        for i in perf:
            line = i.split('\t')
            number = [int(num) for num in line[0][1:-1].split(',')]
            comb.append(number)
            perf_val.append(float(line[1]))

        perf_per_duo = {}

        for i in tqdm(self.list_duo):

            mean_perf = []

            for j in np.arange(len(comb)):
                if i[0] in comb[j] and i[1] in comb[j]:
                    mean_perf.append(perf_val[j])

            if mean_perf:
                perf_per_duo['{var1},{var2}'.format(var1=i[0], var2=i[1])] = np.mean(mean_perf)

        return perf_per_duo

    def counts(self):

        best_var = []
        for i in self.fp.read_only_comb():
            val = i[1:-1].split(',')
            for j in val:
                best_var.append(int(j))
        select_var, counts = np.unique(best_var, return_counts=True)

        count_var = {}
        for i, j in zip(select_var, counts):
            count_var['{}'.format(i)] = j

        return count_var

    def lift(self):

        edge_dict_lift = []

        comb = self.fp.read_only_comb()
        names, id_names = self.fp.read_names()
        perf_duo = self.performance_duo()
        count_var = self.counts()

        for i in tqdm(self.list_duo):

            duo_count = 0

            for j in comb:

                combination = [int(k) for k in j[1:-1].split(',')]

                if i[0] in combination and i[1] in combination:
                    duo_count += 1

            if duo_count > 0:

                local_perf = perf_duo['{var1},{var2}'.format(var1=i[0], var2=i[1])]

                lift = (duo_count / count_var['{}'.format(i[0])]) / (count_var['{}'.format(i[1])] / len(comb))

                if lift > 1:

                    edge_dict_lift.append({'var1': names[i[0]],
                                           'var2': names[i[1]],
                                           'val': lift/local_perf})

        with open('../../var_combination/edge_lift_{}.txt'.format(self.group_name), 'w') as file:
            file.write('Source\tTarget\tWeight\n')
            for i in edge_dict_lift:
                file.write('{var1}\t{var2}\t{val}\n'.format(var1=id_names[i['var1'][1:-1]],
                                                            var2=id_names[i['var2'][1:-1]],
                                                            val=i['val']))

        print('\nDone!')
        os.system("say Done")


if __name__ == "__main__":

    group_name = 'nolb'
    explanans = 163

    freq = Frequencies(group_name=group_name, explanans=explanans)
    freq.edit_file()

    asso = Association(group_name=group_name, explanans=explanans)
    asso.lift()
