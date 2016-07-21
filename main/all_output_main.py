import numpy as np
from itertools import combinations
from collections import OrderedDict
from module.network_trainer import NetworkTrainer  # Version 'pure Python"
# from module.c_network_trainer import NetworkTrainer
from module.save import BackUp
from multiprocessing import Pool
from time import time
from os import path, mkdir


class DataManager(object):
    def __init__(self, file_name="bootstrap", explanans_size=52, explanandum_size=3):

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
    def combinations_samples(cls, n, split_value):

        print("Compute combinations for samples...")
        print("Number of individuals: {}.".format(n))
        print("Split value: {}".format(split_value))
        indexes_list = []
        ind = np.arange(n)

        for i in combinations(ind, split_value):
            indexes_list.append({"learning": i,
                                 "testing": np.setdiff1d(ind, i)})

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
        self.file_name = "{}/cursor.txt".format(self.folder)

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

        self.n_network = 50

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

        n_network = 5
        hidden_layer = 52

        learning_rate = 0.05
        presentation_number = 1000

        data_manager = DataManager()  # Import txt file
        data_manager.format_data()  # Center-reduce input variables and normalize output variables

        n = data_manager.data.shape[0]

        # indexes_list = SamplesCreator.combinations_samples(n=n, split_value=int(0.8*n))

        indexes_list = []

        for i in range(n_network):
            selec = np.random.choice(1000, 800, replace=False)
            not_select = np.setdiff1d(np.arange(1000), selec)
            indexes_list.append({
                'learning': selec,
                'testing': not_select
            })

        np.random.shuffle(indexes_list)

        id_network = 0

        for selected_ind in indexes_list[0:n_network]:

            samples_learning = data_manager.import_data(explanandum=[0, 1, 2],
                                                        individuals=selected_ind['learning'])

            samples_testing = data_manager.import_data(explanandum=[0, 1, 2],
                                                       individuals=selected_ind['testing'])

            kwargs = {"dataset": samples_learning,
                      "test_dataset": samples_testing,
                      "hidden_layer": [hidden_layer],
                      "presentation_number": presentation_number,
                      "learning_rate": learning_rate,
                      "momentum": learning_rate,
                      "id_network": id_network,
                      'ind_learning': selected_ind['learning'],
                      'ind_testing': selected_ind['testing']
                      }

            id_network += 1

            self.kwargs_list.append(kwargs)

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

            results = self.pool.map(self.check_all_variables_in_the_same_time,
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

            results = self.pool.map(self.check_all_variables_in_the_same_time, self.kwargs_list[self.cursor.position:])
            self.back_up.save(results)

        time_spent = self.convert_seconds_to_h_m_s(time() - beginning_time)

        print("Cursor position: {}/{} (time spent: {}).".format(to_do, to_do, time_spent))
        print("********************")

        self.cursor.reset()

        print("End of testing program.")


    @staticmethod
    def check_all_variables_in_the_same_time(kwargs):

        network_trainer = NetworkTrainer()
        network_trainer.create_network(dataset=kwargs['dataset'], hidden_layer=kwargs['hidden_layer'])

        pre_test_error, pre_test_output = network_trainer.test_the_network(kwargs['dataset'])
        pre_test2_error, pre_test2_output = network_trainer.test_the_network(kwargs['test_dataset'])
        #
        # weights = network_trainer.network.weights[0]
        # filename = 'weights/weights_init_{}.txt'.format(kwargs['id_network'])
        # np.savetxt(filename, weights)
        #
        # weights2 = network_trainer.network.weights[1]
        # filename2 = 'weights/weights_init_2_{}.txt'.format(kwargs['id_network'])
        # np.savetxt(filename2, weights2)

        network_trainer.teach_the_network(presentation_number=kwargs['presentation_number'],
                                          dataset=kwargs['dataset'],
                                          learning_rate=kwargs['learning_rate'],
                                          momentum=kwargs['momentum'])

        test_error, test_output = network_trainer.test_the_network(kwargs['dataset'])
        test2_error, test_output2 = network_trainer.test_the_network(kwargs['test_dataset'])

        # weights = network_trainer.network.weights[0]
        # filename = 'weights/weights_test_{}.txt'.format(kwargs['id_network'])
        # np.savetxt(filename, weights)
        #
        # weights2 = network_trainer.network.weights[1]
        # filename2 = 'weights/weights_test_2_{}.txt'.format(kwargs['id_network'])
        # np.savetxt(filename2, weights2)

        output = OrderedDict()
        output['pre_learning'] = np.mean(pre_test_error ** 2)  # errors
        output['post_learning'] = np.mean(test_error ** 2)  # errors
        output['pre_learning_test'] = np.mean(pre_test2_error ** 2)  # errors
        output['post_learning_test'] = np.mean(test2_error ** 2)  # errors
        output['presentation_number'] = kwargs['presentation_number']
        output['hidden_layer'] = kwargs['hidden_layer']
        output['learning_rate'] = kwargs['learning_rate']
        output['momentum'] = kwargs['momentum']
        output['id_network'] = kwargs['id_network']
        output['ind_learning'] = kwargs['ind_learning']
        output['ind_testing'] = kwargs['ind_testing']
        for i in range(len(kwargs["test_dataset"]['x'])):
            output['ind{i}'.format(i=i)] = test_output2[i]

        learn_index = (output['pre_learning'] - output['post_learning']) / output['post_learning']

        test_index = (output['pre_learning_test'] - output['post_learning_test']) / output[
            'post_learning_test']

        output['index_learn'] = 100 * learn_index
        output['index_test'] = 100 * test_index

        kwargs.pop('dataset')

        return output


def parameter_test():

    supervisor = Supervisor(n_workers=6, output_file='results_bootstrap', back_up_frequency=1)

    print("\n*************************")
    print('Preparing kwarg list...')
    print("**************************")

    supervisor.prepare_kwargs_list()

    print("**************************")
    print('Kwarg list ready.')
    print("\n*************************")

    supervisor.launch_test()


if __name__ == '__main__':

    parameter_test()
