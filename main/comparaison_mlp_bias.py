import numpy as np
from collections import OrderedDict
from multiprocessing import Pool
from time import time

from module.network_trainer import NetworkTrainer
from module.network_trainer_bias import NetworkTrainer as NetworkTrainer_bias
from module.save_multiproc import BackUp
from module.cursor import Cursor
from module.data_manager import DataManager, SamplesCreator


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

        hidden_layer_out0 = 30
        hidden_layer_out1 = 52
        hidden_layer_out2 = 52

        learning_rate_out0 = 0.07
        learning_rate_out1 = 0.05
        learning_rate_out2 = 0.05

        presentation_number_out0 = 1000
        presentation_number_out1 = 2000
        presentation_number_out2 = 2000

        data_manager = DataManager(file_name='dataset_290416_3output')  # Import txt file
        data_manager.format_data()  # Center-reduce input variables and normalize output variables

        n = data_manager.data.shape[0]

        indexes_list = SamplesCreator.combinations_samples(n=n, split_value=int(0.65*n))

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
                      "id_network": id_network
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

        ntwk_bias = NetworkTrainer_bias()

        output = OrderedDict()

        for i in [0, 1, 2]:

            learning_dataset = kwargs['dataset_out{}'.format(i)]
            test_dataset = kwargs['test_dataset_out{}'.format(i)]

            output['presentation_number_out{}'.format(i)] = kwargs['presentation_number_out{}'.format(i)]
            output['hidden_layer_out{}'.format(i)] = kwargs['hidden_layer_out{}'.format(i)]
            output['learning_rate_out{}'.format(i)] = kwargs['learning_rate_out{}'.format(i)]
            output['momentum_out{}'.format(i)] = kwargs['momentum_out{}'.format(i)]

            # original MLP without bias on hidden layers

            network_trainer.create_network(dataset=kwargs['dataset_out{}'.format(i)],
                                           hidden_layer=kwargs['hidden_layer_out{}'.format(i)])

            pre_test_error, pre_test_output = network_trainer.test_the_network(learning_dataset)
            pre_test2_error, pre_test2_output = network_trainer.test_the_network(test_dataset)

            network_trainer.teach_the_network(presentation_number=kwargs['presentation_number_out{}'.format(i)],
                                              dataset=kwargs['dataset_out{}'.format(i)],
                                              learning_rate=kwargs['learning_rate_out{}'.format(i)],
                                              momentum=kwargs['momentum_out{}'.format(i)])

            test_error, test_output = network_trainer.test_the_network(learning_dataset)
            test2_error, test_output2 = network_trainer.test_the_network(test_dataset)

            output['pre_learning_out{}'.format(i)] = np.mean(pre_test_error ** 2)
            output['post_learning_out{}'.format(i)] = np.mean(test_error ** 2)
            output['pre_learning_test_out{}'.format(i)] = np.mean(pre_test2_error ** 2)
            output['post_learning_test_out{}'.format(i)] = np.mean(test2_error ** 2)

            learn_index = (output['pre_learning_out{}'.format(i)] - output['post_learning_out{}'.format(i)]) / \
                          output['post_learning_out{}'.format(i)]

            test_index = (output['pre_learning_test_out{}'.format(i)] - output[
                'post_learning_test_out{}'.format(i)]) / output['post_learning_test_out{}'.format(i)]

            output['index_learn_out{}'.format(i)] = 100 * learn_index
            output['index_test_out{}'.format(i)] = 100 * test_index


            # new MLP with bias on hidden layers

            ntwk_bias.create_network(dataset=kwargs['dataset_out{}'.format(i)],
                                     hidden_layer=kwargs['hidden_layer_out{}'.format(i)])

            pre_test_error, pre_test_output = ntwk_bias.test_the_network(learning_dataset)
            pre_test2_error, pre_test2_output = ntwk_bias.test_the_network(test_dataset)

            ntwk_bias.teach_the_network(presentation_number=kwargs['presentation_number_out{}'.format(i)],
                                        dataset=kwargs['dataset_out{}'.format(i)],
                                        learning_rate=kwargs['learning_rate_out{}'.format(i)],
                                        momentum=kwargs['momentum_out{}'.format(i)])

            test_error, test_output = ntwk_bias.test_the_network(learning_dataset)
            test2_error, test_output2 = ntwk_bias.test_the_network(test_dataset)

            output['bias_pre_learning_out{}'.format(i)] = np.mean(pre_test_error ** 2)
            output['bias_post_learning_out{}'.format(i)] = np.mean(test_error ** 2)
            output['bias_pre_learning_test_out{}'.format(i)] = np.mean(pre_test2_error ** 2)
            output['bias_post_learning_test_out{}'.format(i)] = np.mean(test2_error ** 2)

            learn_index = (output['bias_pre_learning_out{}'.format(i)] - output['bias_post_learning_out{}'.format(i)]) / \
                          output['bias_post_learning_out{}'.format(i)]

            test_index = (output['bias_pre_learning_test_out{}'.format(i)] - output[
                'bias_post_learning_test_out{}'.format(i)]) / output['bias_post_learning_test_out{}'.format(i)]

            output['bias_index_learn_out{}'.format(i)] = 100 * learn_index
            output['bias_index_test_out{}'.format(i)] = 100 * test_index

        output['ind_learning'] = kwargs['ind_learning']
        output['ind_testing'] = kwargs['ind_testing']
        output['id_network'] = kwargs['id_network']

        kwargs.pop('dataset_out2')

        return output


def parameter_test():

    supervisor = Supervisor(n_workers=6, output_file='results_mlp_bias', back_up_frequency=100)

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