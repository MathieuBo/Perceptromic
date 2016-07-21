import numpy as np
from itertools import combinations
from multiprocessing import Pool
from module.save import Database, BackUp
import pandas
import matplotlib.pyplot as plt
from matplotlib import colors


class Statistician(object):

    def __init__(self):
        self.database = Database('analysis_comb_3')

    def compute(self, v):
        return self.database.read_column(column_name='e_mean', v=v)[0]

    def combinations_of_selected_indexes(self, indexes, size):

        return [i for i in combinations(indexes, size)]

    def analyse_combinations(self, pool, indexes):

        comb_var = self.combinations_of_selected_indexes(indexes=indexes, size=3)

        result = np.zeros(len(comb_var))
        result[:] = pool.map(self.compute, comb_var)

        return comb_var, result
        #return np.mean(result), np.std(result)/np.sqrt(len(usual))


class DataClassifier(Statistician):

    def __init__(self):

        Statistician.__init__(self)

        self.best = None
        self.worst = None

        self.test_dic = None

        self.names = "actim;ss.inac;ss.invest;ss.loc;ss.soc.beh;ss.four.leg;ss.lying.down;ss.slump;" \
                     "ss.enf;ss.eno;ss.ext;ss.soc;ss.sol;ss.corr;wb.syn.put;wb.syn.sn;db.syn.sn;db.syn.putr;" \
                     "db.syn.putc;db.syn.cd;db.synO1.sn;db.synO1.putr;db.synO1.putc;db.synO1.cd;sysy.fe;sysy.cu;" \
                     "sysy.zn;sysy.ca;sysy.se;sysy.mn;bm.blood;bm.plasma;bm.serum;bm.csf;h.syn.wm;h.syn.cd.h;" \
                     "h.syn.put.dl;h.syn.put.dm;h.syn.put.vl;h.syn.put.vm;h.syn.gpe;h.syn.gpi;h.syn.ctx.er;" \
                     "h.syn.ctx.temp;h.syn.ctx.ins;h.syn.ctx.sm;h.syn.ctx.m;h.syn.ctx.cg;wb.syn.putc.hmw;wb.ub.mono;" \
                     "wb.ub.n;wb.ub.tot".split(";")

        self.var_group = {
            'behavior': {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 'color': 'dodgerblue'},
            'dotblot': {'index': [16, 17, 18, 19, 20, 21, 22, 23], "color": 'sage'},
            'synchrotron': {'index': [24, 25, 26, 27, 28, 29], 'color': 'darkorange'},
            'histology': {'index': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], 'color': 'steelblue'},
            'wb': {"index": [14, 15, 48, 49, 50, 51], 'color': 'teal'},
            'biomarkers': {'index': [30, 31, 32, 33], 'color': 'indianred'}
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

        selection = int(len(sorted_list)/100)  # Selection of the 1% best

        print('Total considered combinations : {}'.format(selection))

        best_comb = sorted_list[:selection]

        best_values =[]
        for i in best_comb:
            best_values.append(self.test_dic[i])

        best_mean = np.mean(best_values)
        best_sem = np.std(best_values)/selection

        print("overvall performance of best combinations : {} +/- {}\n".format(best_mean, best_sem))

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
        plt.savefig('graph_var_comb/{}.eps'.format(table_name), dpi=300)
        plt.show()

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
        worst_sem = np.std(worst_values)/selection

        print("overvall performance of worst combinations : {} +/- {}\n".format(worst_mean, worst_sem))

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

    def __init__(self):

        Statistician.__init__(self)

    def overall_mean(self, pool, indexes):

        print('**********************')
        print('Usual var index are : {}'.format(indexes))

        comb_var, results = self.analyse_combinations(pool, indexes)

        print('Number of combinations used : {}'.format(len(comb_var)))

        mean = np.mean(results)
        sem = np.std(results)/np.sqrt(len(comb_var))

        print('Overall mean = {} +/- {}'.format(mean, sem))
        print('**********************')


if __name__ == "__main__":

    import time
    print(time.strftime("%d/%m/%Y"))
    print(time.strftime("%H:%M:%S"))
    pool = Pool(processes=8)

    indexes_usual = np.asarray([0,14,15,16,17,18,19,24,30,31,32,33,35,36,37,38,39,40,41,42,46,48,49,50,51])
    indexes_unusual = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22,23,25,26,27,28,29,34,43,44,45,47])
    indexes_syn = np.asarray([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,48])
    indexes_nosyn = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,13,24,25,26,27,28,29,49,50,51])

    # s = SimpleAnalyse()
    # s.overall_mean(pool=pool, indexes=indexes_usual)

    d = DataClassifier()

    d.classification(indexes=indexes_usual, table_name='usual', pool=pool)
    d.get_best_worst(table_name='usual')
    d.classification(indexes=indexes_unusual, table_name='unusual', pool=pool)
    d.get_best_worst(table_name='unusual')
    d.classification(indexes=indexes_syn, table_name='Syn_related', pool=pool)
    d.get_best_worst(table_name='Syn_related')
    d.classification(indexes=indexes_nosyn, table_name='Not_syn_related', pool=pool)
    d.get_best_worst(table_name='Not_syn_related')