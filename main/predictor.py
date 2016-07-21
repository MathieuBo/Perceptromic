import numpy as np
import matplotlib.pyplot as plt
from module.save import Database, BackUp
from collections import OrderedDict
from module.network_trainer import NetworkTrainer
from single_output_main import DataManager


class Selector:

    def __init__(self, filename):

        self.filename = filename
        self.db = Database(self.filename)

    def select_networks(self):

        selection_param = OrderedDict()
        selection_param["out0"] = {'index': 2000.0, 'value': 0.02}
        selection_param["out1"] = {'index': 7000.0, 'value': 0.01}
        selection_param["out2"] = {'index': 10000.0, 'value': 0.002}

        selected_networks = OrderedDict()

        for i, out in enumerate(selection_param):

            test_index = self.db.read_column("index_test_out{}".format(i))
            test_index = [float(i) for i in test_index]

            test_value = self.db.read_column("post_learning_test_out{}".format(i))
            test_value = [float(i) for i in test_value]

            network_list = []

            for j in range(len(test_index)):
                if test_index[j] > selection_param[out]['index'] and test_value[j] < selection_param[out]['value']:
                    network_list.append(j)

            selected_networks['out{}'.format(i)] = network_list

        return selected_networks

    def network_combination(self):

        selected_networks = self.select_networks()

        combinations = []

        for i in selected_networks['out0']:
            for j in selected_networks['out1']:
                for k in selected_networks['out2']:
                    test = i, j, k
                    combinations.append(test)

        return combinations

    def matrix_builder(self):

        comb = self.network_combination()

        weight_matrices = OrderedDict()

        for i in range(len(comb)):

            matrix_list = []

            for j, k in enumerate(comb[i]):

                layer1 = np.loadtxt('weights_single/weights_test_{id}_out{output}.txt'.format(id=k, output=j))
                matrix_list.append(layer1)

                layer2 = np.loadtxt('weights_single/weights_test_2_{id}_out{output}.txt'.format(id=k, output=j))
                matrix_list.append(layer2)

            weight_matrices[comb[i]] = matrix_list

        return weight_matrices


class Supervisor:

    def __init__(self, filename):

        self.selector = Selector(filename=filename)

        self.network_param = {
            'out0': 30,
            'out1': 52,
            'out2': 52
        }

    @staticmethod
    def data_import():

        data_manager = DataManager()
        data_manager.format_data()

        dataset = []

        for i in [0, 1, 2]:
            dataset.append(data_manager.import_data(explanandum=[i]))

        return dataset

    def predictor(self):

        network_trainer = NetworkTrainer()

        dataset = self.data_import()

        weights_matrices = self.selector.matrix_builder()

        for matrices in weights_matrices:

            ind0 = []
            ind1 = []
            ind2 = []

            for j in [0, 1, 2]:

                network_trainer.create_network(dataset=dataset[j], hidden_layer=[self.network_param['out{}'.format(j)]])

                network_trainer.network.weights = weights_matrices[matrices][j]

                test2_error, test_output2 = network_trainer.test_the_network(dataset[j])

                print(test2_error)
                print(test_output2)


class WeightAnalyst:

    def __init__(self, filename):

        self.filename = filename

    def graphist(self):

        selector = Selector(filename=self.filename)

        selected_networks = selector.select_networks()

        for i,key in enumerate(selected_networks):

            net0 = np.loadtxt('weights_single/weights_test_{id}_out{output}.txt'.format(id=selected_networks[key][0], output=i))
            net1 = np.loadtxt('weights_single/weights_test_{id}_out{output}.txt'.format(id=selected_networks[key][1], output=i))
            # net2 = np.loadtxt('weights_single/weights_test_{id}_out{output}.txt'.format(id=selected_networks[key][2], output=i))

            net0 = np.where(np.absolute(net0) > 0.20, net0, 0)
            net1 = np.where(np.absolute(net1) > 0.20, net1, 0)

            plt.subplot(131)
            plt.title('Best network')
            plt.imshow(net0, interpolation='nearest', cmap='RdBu', vmax=0.35, vmin=-0.35)

            plt.subplot(132)
            plt.title('Second  best network')
            plt.imshow(net1, interpolation='nearest', cmap='RdBu', vmax=0.35, vmin=-0.35)

            # plt.subplot(133)
            # plt.imshow(net2, interpolation='nearest', cmap='RdBu', vmax=0.35, vmin=-0.35)
            plt.colorbar()

            diff = (net0-net1)**2
            print("\n******************")
            print("Mean square error for ouput{}".format(i))
            print(np.mean(diff))

            plt.subplot(133)
            plt.imshow(diff, interpolation='nearest', cmap="hot")
            plt.title('Square difference \nbetween the two matrices')
            plt.colorbar()
            plt.show()

if __name__ == '__main__':

    # s = Supervisor(filename='results_single_output')
    #
    # s.predictor()

    analyst = WeightAnalyst(filename="results_single_output")
    analyst.graphist()