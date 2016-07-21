import numpy as np
from itertools import combinations
from collections import OrderedDict
from module.network_trainer import NetworkTrainer  # Version 'pure Python"
# from module.c_network_trainer import NetworkTrainer
from module.save import BackUp, Database
from multiprocessing import Pool
from time import time
from os import path, mkdir


class DataManager(object):
    def __init__(self, file_name="dataset_290416_3output", explanans_size=52, explanandum_size=3):

        self.folder_path = "data"
        self.file_path = "{}/{}.txt".format(self.folder_path, file_name)
        self.explanans_size = explanans_size
        self.explanandum_size = explanandum_size
        self.data = self.import_txt()

    def import_txt(self):

        print("Import txt file.")

        data = np.loadtxt(self.file_path)
        return data

    def format_data(self):

        # Center reduce for explanans, normalize for explanandum

        data = np.zeros(self.data.shape[0], dtype=[('x', float, self.explanans_size),
                                                   ('y', float, self.explanandum_size)])
        data["x"] = Format.center_reduce(self.data[:, :self.explanans_size])
        data["y"] = Format.normalize(self.data[:, self.explanans_size:])

        self.data = data

    def import_data(self, explanans=None, explanandum=None, individuals=None):

        # Select piece of data

        if explanans is None:
            explanans = np.arange(self.explanans_size)

        if explanandum is None:
            explanandum = np.arange(self.explanandum_size)

        if individuals is None:
            individuals = np.arange(self.data.shape[0])

        data = np.zeros(len(individuals), dtype=[('x', float, len(explanans)),
                                                 ('y', float, len(explanandum))])

        data["x"] = self.data['x'][np.asarray(individuals)][:, explanans]

        if len(explanandum) == 1:
            data["y"] = self.data['y'][np.asarray(individuals)][:, np.asarray(explanandum)].T

        else:
            data["y"] = self.data['y'][np.asarray(individuals)][:, np.asarray(explanandum)]

        return data


class SamplesCreator(object):

    @classmethod
    def combinations_samples(cls, n):

        print("Compute combinations for samples...")
        print("Number of individuals: {}.".format(n))
        print("6 ind for learning, 4 ind for testing, 3 ind for validation")
        indexes_list = []
        ind = np.arange(n)

        val = np.random.choice(ind, 3, replace=False)
        remaining_ind = np.setdiff1d(ind, val)

        ctrl_index = []
        sick_index = []

        for i in range(len(remaining_ind)):
            if remaining_ind[i] < 7:
                ctrl_index.append(remaining_ind[i])
            elif remaining_ind[i] >= 7:
                sick_index.append(remaining_ind[i])
            else:
                raise Exception('Error combination split')

        ctrl_comb = [i for i in combinations(ctrl_index, 3)]
        sick_comb = [i for i in combinations(sick_index, 3)]

        for i in ctrl_comb:
            for j in sick_comb:
                indexes_list.append({'learning': np.concatenate((i, j)),
                                    'testing': np.concatenate((np.setdiff1d(ctrl_index, i),
                                                              np.setdiff1d(sick_index, j))),
                                     'validation': val})

        print("Done.")

        return indexes_list


class Format(object):

    @classmethod
    def normalize(cls, data, new_range=1, new_min=-0.5):
        if len(data.shape) == 1:

            vmin, vmax = data.min(), data.max()
            formatted_data = new_range * (data - vmin) / (vmax - vmin) + new_min

        else:

            formatted_data = data.copy()
            for i in range(data.shape[1]):
                vmin, vmax = data[:, i].min(), data[:, i].max()
                formatted_data[:, i] = new_range * (data[:, i] - vmin) / (vmax - vmin) + new_min

        return formatted_data

    @classmethod
    def center_reduce(cls, data):

        if len(data.shape) == 1:

            mean, std = np.mean(data), np.std(data)
            if std != 0:
                formatted_data = (data - mean) / std
            else:
                formatted_data = (data - mean)

        else:
            formatted_data = np.zeros(data.shape)
            for i in range(data.shape[1]):
                mean, std = np.mean(data[:, i]), np.std(data[:, i])
                if std != 0:
                    formatted_data[:, i] = 2 * (data[:, i] - mean) / std
                else:
                    formatted_data[:, i] = 2 * (data[:, i] - mean)

        return formatted_data


class Cursor(object):

    def __init__(self):
        self.position = 0
        self.folder = "tmp"
        self.file_name = "{}/cursor_single_output.txt".format(self.folder)

    def retrieve_position(self):

        if path.exists(self.file_name):

            f = open(self.file_name, 'r')
            f_content = f.read()
            f.close()

            if f_content == '':

                self.position = 0
            else:

                try:
                    self.position = int(f_content)
                except:
                    self.position = 0
        else:
            if not path.exists(self.folder):
                mkdir(self.folder)
            self.position = 0

    def save_position(self):

        f = open(self.file_name, "w")
        f.write(str(self.position))
        f.close()

    def reset(self):

        f = open(self.file_name, "w")
        f.write(str(0))
        f.close()

        self.position = 0


class Supervisor:

    def __init__(self, n_workers, output_file, back_up_frequency=10000):

        self.n_network = 80

        self.pool = Pool(processes=n_workers)

        self.back_up = BackUp(database_name=output_file)
        self.back_up_fq = back_up_frequency

        self.kwargs_list = []

        self.cursor = Cursor()

    @staticmethod
    def convert_seconds_to_h_m_s(seconds):

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def prepare_kwargs_list(self):

        hidden_layer_out0 = 30
        hidden_layer_out1 = 52
        hidden_layer_out2 = 52

        learning_rate_out0 = 0.07
        learning_rate_out1 = 0.05
        learning_rate_out2 = 0.05

        presentation_number_out0 = 1000
        presentation_number_out1 = 2000
        presentation_number_out2 = 2000

        data_manager = DataManager()  # Import txt file
        data_manager.format_data()  # Center-reduce input variables and normalize output variables

        n = data_manager.data.shape[0]

        indexes_list = SamplesCreator.combinations_samples(n=n)
        print(len(indexes_list))

        np.random.shuffle(indexes_list)

        id_network = 0

        for selected_ind in indexes_list[0:self.n_network]:

            samples_learning_out0 = data_manager.import_data(explanandum=[0],
                                                             individuals=selected_ind['learning'])
            samples_testing_out0 = data_manager.import_data(explanandum=[0],
                                                            individuals=selected_ind['testing'])

            samples_learning_out1 = data_manager.import_data(explanandum=[1],
                                                             individuals=selected_ind['learning'])
            samples_testing_out1 = data_manager.import_data(explanandum=[1],
                                                            individuals=selected_ind['testing'])

            samples_learning_out2 = data_manager.import_data(explanandum=[2],
                                                             individuals=selected_ind['learning'])
            samples_testing_out2 = data_manager.import_data(explanandum=[2],
                                                            individuals=selected_ind['testing'])

            kwargs = {"dataset_out0": samples_learning_out0,
                      "test_dataset_out0": samples_testing_out0,
                      "dataset_out1": samples_learning_out1,
                      "test_dataset_out1": samples_testing_out1,
                      "dataset_out2": samples_learning_out2,
                      "test_dataset_out2": samples_testing_out2,
                      "hidden_layer_out0": [hidden_layer_out0],
                      "hidden_layer_out1": [hidden_layer_out1],
                      "hidden_layer_out2": [hidden_layer_out2],
                      "presentation_number_out0": presentation_number_out0,
                      "presentation_number_out1": presentation_number_out1,
                      "presentation_number_out2": presentation_number_out2,
                      "learning_rate_out0": learning_rate_out0,
                      "learning_rate_out1": learning_rate_out1,
                      "learning_rate_out2": learning_rate_out2,
                      "momentum_out0": learning_rate_out0,
                      "momentum_out1": learning_rate_out1,
                      "momentum_out2": learning_rate_out2,
                      "ind_learning": selected_ind['learning'],
                      "ind_testing": selected_ind['testing'],
                      "id_network": id_network,
                      'validation_set': selected_ind['validation']
                      }

            self.kwargs_list.append(kwargs)

            id_network += 1

    def launch_test(self):

        """
        Require a list of arguments
        :return: None
        """

        if not self.kwargs_list:
            raise Exception("Before beginning testing, arguments should be added to the 'kwargs' list by calling "
                            "method 'fill_kwargs_list'.")

        beginning_time = time()

        print("********************")

        self.cursor.retrieve_position()

        to_do = len(self.kwargs_list)

        print("Begin testing.")

        while self.cursor.position + self.back_up_fq < to_do:
            time_spent = self.convert_seconds_to_h_m_s(time() - beginning_time)

            print("Cursor position: {}/{} (time spent: {}).".format(self.cursor.position, to_do, time_spent))
            print("********************")

            results = self.pool.map(self.check_single_output,
                                    self.kwargs_list[self.cursor.position:self.cursor.position + self.back_up_fq])

            self.back_up.save(results)

            self.cursor.position += self.back_up_fq

            self.cursor.save_position()

        if self.cursor.position + self.back_up_fq == (to_do - 1):

            pass

        else:
            time_spent = self.convert_seconds_to_h_m_s(time() - beginning_time)

            print("Cursor position: {}/{} (time spent: {}).".format(self.cursor.position, to_do, time_spent))
            print("********************")

            results = self.pool.map(self.check_single_output, self.kwargs_list[self.cursor.position:])
            self.back_up.save(results)

        time_spent = self.convert_seconds_to_h_m_s(time() - beginning_time)

        print("Cursor position: {}/{} (time spent: {}).".format(to_do, to_do, time_spent))
        print("********************")

        self.cursor.reset()

        print("End of testing program.")


    @staticmethod
    def check_single_output(kwargs):

        network_trainer = NetworkTrainer()

        output = OrderedDict()

        ind0 = []
        ind1 = []
        ind2 = []
        ind3 = []

        for i in [0, 1, 2]:

            network_trainer.create_network(dataset=kwargs['dataset_out{}'.format(i)], hidden_layer=kwargs['hidden_layer_out{}'.format(i)])

            learning_dataset = kwargs['dataset_out{}'.format(i)]
            test_dataset = kwargs['test_dataset_out{}'.format(i)]

            pre_test_error, pre_test_output = network_trainer.test_the_network(learning_dataset)
            pre_test2_error, pre_test2_output = network_trainer.test_the_network(test_dataset)

            network_trainer.teach_the_network(presentation_number=kwargs['presentation_number_out{}'.format(i)],
                                              dataset=kwargs['dataset_out{}'.format(i)],
                                              learning_rate=kwargs['learning_rate_out{}'.format(i)],
                                              momentum=kwargs['momentum_out{}'.format(i)])

            test_error, test_output = network_trainer.test_the_network(learning_dataset)
            test2_error, test_output2 = network_trainer.test_the_network(test_dataset)

            weights = network_trainer.network.weights[0]
            filename = 'weights_predictor/weights_test_{id}_out{output}.txt'.format(id=kwargs['id_network'], output=i)
            np.savetxt(filename, weights)

            weights2 = network_trainer.network.weights[1]
            filename2 = 'weights_predictor/weights_test_2_{id}_out{output}.txt'.format(id=kwargs['id_network'], output=i)
            np.savetxt(filename2, weights2)

            output['pre_learning_out{}'.format(i)] = np.mean(pre_test_error ** 2)
            output['post_learning_out{}'.format(i)] = np.mean(test_error ** 2)
            output['pre_learning_test_out{}'.format(i)] = np.mean(pre_test2_error ** 2)
            output['post_learning_test_out{}'.format(i)] = np.mean(test2_error ** 2)

            ind0.append(test_output2[0])
            ind1.append(test_output2[1])
            ind2.append(test_output2[2])
            ind3.append(test_output2[3])

            # for j in range(len(kwargs["test_dataset_out{}".format(i)]['x'])):
            #     output['ind{j}'.format(j=j)] = test_output2[j]

            output['presentation_number_out{}'.format(i)] = kwargs['presentation_number_out{}'.format(i)]
            output['hidden_layer_out{}'.format(i)] = kwargs['hidden_layer_out{}'.format(i)]
            output['learning_rate_out{}'.format(i)] = kwargs['learning_rate_out{}'.format(i)]
            output['momentum_out{}'.format(i)] = kwargs['momentum_out{}'.format(i)]

            learn_index = (output['pre_learning_out{}'.format(i)] - output['post_learning_out{}'.format(i)]) / \
                          output['post_learning_out{}'.format(i)]

            test_index = (output['pre_learning_test_out{}'.format(i)] - output[
                'post_learning_test_out{}'.format(i)]) / output['post_learning_test_out{}'.format(i)]

            output['index_learn_out{}'.format(i)] = 100 * learn_index
            output['index_test_out{}'.format(i)] = 100 * test_index

        output['ind0'] = ind0
        output['ind1'] = ind1
        output['ind2'] = ind2
        output['ind3'] = ind3

        output['ind_learning'] = kwargs['ind_learning']
        output['ind_testing'] = kwargs['ind_testing']
        output['id_network'] = kwargs['id_network']
        output['validation_set'] = kwargs['validation_set']

        kwargs.pop('dataset_out2')

        return output


def parameter_test():

    supervisor = Supervisor(n_workers=6, output_file='results_combinator', back_up_frequency=100)

    print("\n*************************")
    print('Preparing kwarg list...')
    print("**************************")

    supervisor.prepare_kwargs_list()

    print("**************************")
    print('Kwarg list ready.')
    print("\n*************************")

    supervisor.launch_test()


class Selector:

    def __init__(self, filename):

        self.filename = filename
        self.db = Database(self.filename)

    def select_networks(self):

        selection_param = OrderedDict()
        selection_param["out0"] = {'index': 450.0, 'value': 0.03}
        selection_param["out1"] = {'index': 2400.0, 'value': 0.008}
        selection_param["out2"] = {'index': 950.0, 'value': 0.03}

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

    def matrix_builder(self):

        selected_networks = self.select_networks()

        weight_matrices = dict()

        for key in selected_networks:

            matrix = []

            layer1 = np.loadtxt('weights_predictor/weights_test_{id}_{output}.txt'.format(id=selected_networks[key][0], output=key))
            matrix.append(layer1)

            layer2 = np.loadtxt('weights_predictor/weights_test_2_{id}_{output}.txt'.format(id=selected_networks[key][0], output=key))
            matrix.append(layer2)

            weight_matrices['{}'.format(key)] = matrix

        return selected_networks, weight_matrices

    def convert_group(self, results):

        conv_results = np.zeros((len(results), 3))

        for i, result in enumerate(results):
            remove_charac = str.maketrans("", "", "[]")
            remove_coma = str.maketrans("", "", ",")
            result = result.translate(remove_coma).translate(remove_charac).split()
            result = [int(j) for j in result]
            conv_results[i] = result

        return conv_results

    def test_new_network(self):

        print('NETWORK COMBINATOR : VALIDATION STEP\n')

        network_trainer = NetworkTrainer()

        data_manager = DataManager()
        data_manager.format_data()

        validation_set = list(self.convert_group(self.db.read_column('validation_set'))[0])
        validation_set = [int(i) for i in validation_set]

        print('\n*************************')
        print('Ind in the validation set : {}'. format(validation_set))

        selected_networks, matrix = self.matrix_builder()

        print('\n*************************')
        print('Selected network for output 0 : {}'.format(selected_networks['out0']))
        print('Selected network for output 1 : {}'.format(selected_networks['out1']))
        print('Selected network for output 2 : {}'.format(selected_networks['out2']))

        network_param = [30, 52, 52]

        for i in [0, 1, 2]:

            data = data_manager.import_data(explanandum=[i], individuals=validation_set)

            network_trainer.create_network(dataset=data, hidden_layer=[network_param[i]])

            network_trainer.network.weights = matrix['out{}'.format(i)]

            test2_error, test_output2 = network_trainer.test_the_network(data)

            print('\n*************************')
            print('Results for output : {}'.format(i))
            print('Mean square error on 3 validation ind : {}'.format(np.mean(test2_error)**2))

    def test_var_strength(self):

        print('NETWORK COMBINATOR : VALIDATION STEP\n')

        network_trainer = NetworkTrainer()

        data_manager = DataManager()
        data_manager.format_data()

        validation_set = list(self.convert_group(self.db.read_column('validation_set'))[0])
        validation_set = [int(i) for i in validation_set]

        print('\n*************************')
        print('Ind in the validation set : {}'.format(validation_set))

        selected_networks, matrix = self.matrix_builder()

        print('\n*************************')
        print('Selected network for output 0 : {}'.format(selected_networks['out0']))
        print('Selected network for output 1 : {}'.format(selected_networks['out1']))
        print('Selected network for output 2 : {}'.format(selected_networks['out2']))

        network_param = [30, 52, 52]

        for i in [0, 1, 2]:

            data = data_manager.import_data(explanandum=[i], individuals=validation_set)

            network_trainer.create_network(dataset=data, hidden_layer=[network_param[i]])

            network_trainer.network.weights = matrix['out{}'.format(i)]

            print(network_trainer.network.weights[0].shape)

            output = np.zeros((53, 2))

            test2_error, test_output2 = network_trainer.test_the_network(data)

            output[0, 0] = np.mean(test2_error)**2
            output[0, 1] = np.std(test2_error)**2

            for j in range(52):

                network_trainer.network.weights[0][j, :] = network_trainer.network.weights[0][j, :]*0

                test2_error, test_output2 = network_trainer.test_the_network(data)

                output[j+1, 0] = np.mean(test2_error) ** 2
                output[j+1, 1] = np.std(test2_error) ** 2

            filename = 'var_strength/perf_out{output}.txt'.format(output=i)
            np.savetxt(filename, output)

if __name__ == '__main__':

    # parameter_test()

    # selector = Selector(filename='results_combinator')
    # selector.test_new_network()

    selector = Selector(filename='results_combinator')
    selector.test_var_strength()