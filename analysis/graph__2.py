import numpy as np
from module.save import Database
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DbToImg:

    def __init__(self, filename):

        self.filename = filename
        self.db = Database(self.filename)

    def convert_data(self, results):

        conv_results = np.zeros((len(results), 3))

        for i, result in enumerate(results):
            remove_charac = str.maketrans("", "", "[]")
            result = result.translate(remove_charac).split()
            result = [float(j) for j in result]
            conv_results[i] = result

        return conv_results

    def convert_data_single(self, results):

        conv_results = np.zeros((len(results), 3))

        for i, result in enumerate(results):
            remove_charac = str.maketrans("", "", "[]")
            remove_coma = str.maketrans("", "", ",")
            result = result.translate(remove_coma).translate(remove_charac).split()
            result = [float(j) for j in result]
            conv_results[i] = result

        return conv_results

    def convert_group(self, results):

        conv_results = np.zeros((len(results), 5))

        for i, result in enumerate(results):
            remove_charac = str.maketrans("", "", "[]")
            remove_coma = str.maketrans("", "", ",")
            result = result.translate(remove_coma).translate(remove_charac).split()
            result = [float(j) for j in result]
            conv_results[i] = result

        return conv_results

    def graph(self):

        index_test = self.db.read_column('index_test')
        index_test = [float(i) for i in index_test]

        plt.hist(index_test, bins=50)
        plt.show()

        results_ind0 = self.convert_data(self.db.read_column(column_name='ind0'))
        results_ind1 = self.convert_data(self.db.read_column(column_name='ind1'))
        results_ind2 = self.convert_data(self.db.read_column(column_name='ind2'))

        test_index = self.db.read_column(column_name='index_test')
        test_index = [float(i) for i in test_index]

        test_value = self.db.read_column(column_name='post_learning_test')
        test_value = [float(i) for i in test_value]

        bad_networks = []

        for i in range(len(results_ind0)):
            if test_index[i] < 300.0 or test_value[i] > 0.04:
                bad_networks.append(i)

        results_ind0 = np.delete(results_ind0, bad_networks, axis=0)
        results_ind1 = np.delete(results_ind1, bad_networks, axis=0)
        results_ind2 = np.delete(results_ind2, bad_networks, axis=0)

        group = self.convert_data_single(self.db.read_column('ind_testing'))
        group = np.delete(group, bad_networks, axis=0)

        id_network = self.db.read_column('id_network')
        id_network = np.delete(id_network, bad_networks)

        ctrl_ind = [0, 1, 2, 3, 4, 5, 6]
        park_ind = [7, 8, 9, 10, 11, 12]

        group_conv = []

        for i in range(len(group)):
            for j in group[i]:

                if j in ctrl_ind:
                    anwser_ctrl = "#00bfff"
                    group_conv.append(anwser_ctrl)
                elif j in park_ind:
                    answer_park = "#ff0000"
                    group_conv.append(answer_park)
                else:
                    raise Exception('Error: Ind does not belong to ctrl or park')

        conv_results = np.concatenate((results_ind0, results_ind1, results_ind2), axis=0)
        np.savetxt('predicted_points.txt', conv_results)

        real_data = np.loadtxt('data.txt')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(conv_results)):
            ax.scatter(xs=conv_results[i, 0], ys=conv_results[i, 1], zs=conv_results[i, 2], c=group_conv[i])
            label = '{}'.format(id_network[i//3])
            ax.text(conv_results[i, 0], conv_results[i, 1], conv_results[i, 2], label)

        ax.scatter(xs=real_data[:, 0], ys=real_data[:, 1], zs=real_data[:, 2], marker='^', c='k', s=30)

        ax.set_xlabel('Output_1')
        ax.set_ylabel('Output_2')
        ax.set_zlabel('Output_3')
        plt.show()

    def graph_single_output(self):

        test_0 = self.db.read_column('index_test_out0')
        test_0 = [float(i) for i in test_0]

        test_1 = self.db.read_column('index_test_out1')
        test_1 = [float(i) for i in test_1]

        test_2 = self.db.read_column('index_test_out2')
        test_2 = [float(i) for i in test_2]

        plt.subplot(131)
        plt.hist(test_0, bins=50)
        plt.subplot(132)
        plt.hist(test_1, bins=50)
        plt.subplot(133)
        plt.hist(test_2, bins=50)
        plt.show()

        results_ind0 = self.convert_data_single(self.db.read_column(column_name='ind0'))
        results_ind1 = self.convert_data_single(self.db.read_column(column_name='ind1'))
        results_ind2 = self.convert_data_single(self.db.read_column(column_name='ind2'))
        results_ind3 = self.convert_data_single(self.db.read_column(column_name='ind3'))
        results_ind4 = self.convert_data_single(self.db.read_column(column_name='ind4'))

        error = dict()

        for i in range(3):
            index = self.db.read_column(column_name='post_learning_test_out{}'.format(i))
            index = [float(i) for i in index]
            error['out{}'.format(i)] = index

        test_index = dict()

        for i in range(3):
            index = self.db.read_column(column_name='index_test_out{}'.format(i))
            index = [float(i) for i in index]
            test_index['out{}'.format(i)] = index

        bad_networks = []

        for i in range(len(results_ind0)):
            if error['out0'][i] > 0.06 or error['out1'][i] > 0.06 or error['out2'][i] > 0.06:
                bad_networks.append(i)

            # if test_index['out0'][i] < 500.0 or test_index['out1'][i] <500.0 or test_index['out2'][i] < 500.0:
            #     bad_networks.append(i)

        results_ind0 = np.delete(results_ind0, bad_networks, axis=0)
        results_ind1 = np.delete(results_ind1, bad_networks, axis=0)
        results_ind2 = np.delete(results_ind2, bad_networks, axis=0)
        results_ind3 = np.delete(results_ind3, bad_networks, axis=0)
        results_ind4 = np.delete(results_ind4, bad_networks, axis=0)

        group = self.convert_group(self.db.read_column('ind_testing'))
        group = np.delete(group, bad_networks, axis=0)

        new_group = []

        for i in range(len(group)):
            print(group[i])
            for j in group[i]:
                new_group.append(int(j))

        id_network = self.db.read_column('id_network')
        id_network = np.delete(id_network, bad_networks)

        ctrl_ind = [0, 1, 2, 3, 4, 5, 6]
        park_ind = [7, 8, 9, 10, 11, 12]

        group_conv = []

        for i in range(len(group)):

            for j in group[i]:

                if j in ctrl_ind:
                    anwser_ctrl = "#00bfff"
                    group_conv.append(anwser_ctrl)
                elif j in park_ind:
                    answer_park = "#ff0000"
                    group_conv.append(answer_park)
                else:
                    raise Exception('Error: Ind does not belong to ctrl or park')

        conv_results = np.concatenate((results_ind0, results_ind1, results_ind2, results_ind3, results_ind4), axis=0)
        np.savetxt('predicted_points_single.txt', conv_results)

        real_data = np.loadtxt('data.txt')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(conv_results)):
            ax.scatter(xs=conv_results[i, 0], ys=conv_results[i, 1], zs=conv_results[i, 2], c=group_conv[i])
            # label = '{}'.format(id_network[i//3])
            # ax.text(conv_results[i, 0], conv_results[i, 1], conv_results[i, 2], label)

            ax.text(conv_results[i, 0], conv_results[i, 1], conv_results[i, 2], new_group[i])

        for i in range(0, 7):
            ax.scatter(xs=real_data[i, 0], ys=real_data[i, 1], zs=real_data[i, 2], marker='^', c='b', s=50)

        for i in range(7, 13):
            ax.scatter(xs=real_data[i, 0], ys=real_data[i, 1], zs=real_data[i, 2], marker='^', c='r', s=50)

        for i in range(len(real_data)):
            ax.text(real_data[i, 0], real_data[i, 1], real_data[i, 2], i)

        ax.set_xlabel('Output_1')
        ax.set_ylabel('Output_2')
        ax.set_zlabel('Output_3')
        plt.show()

if __name__ == '__main__':

    # manager = DbToImg(filename="results_all_output")
    # manager.graph()

    manager = DbToImg(filename="results_single_output_split8")
    manager.graph_single_output()
