import numpy as np
from collections import OrderedDict
from module.c_network_trainer import NetworkTrainer
from module.save import BackUp
from multiprocessing import Pool
from time import time
from os import path, mkdir


class Cursor(object):

    def __init__(self, id_job):
        self.position = 0
        self.folder = "tmp"
        self.file_name = "{folder}/cursor_combinations_{id_job}.txt".format(folder=self.folder, id_job=id_job)

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

    def __init__(self, n_workers, output_file, back_up_frequency, kwargs_list, id_job):

        self.n_network = 50

        self.pool = Pool(processes=n_workers)

        self.back_up = BackUp(database_name=output_file)
        self.back_up_fq = back_up_frequency

        self.kwargs_list = kwargs_list

        self.cursor = Cursor(id_job=id_job)

    @staticmethod
    def convert_seconds_to_h_m_s(seconds):

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

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

        network_trainer.teach_the_network(presentation_number=kwargs['presentation_number'],
                                          dataset=kwargs['dataset'],
                                          learning_rate=kwargs['learning_rate'],
                                          momentum=kwargs['momentum'])

        test_error, test_output = network_trainer.test_the_network(kwargs['dataset'])
        test2_error, test_output2 = network_trainer.test_the_network(kwargs['test_dataset'])

        output = OrderedDict()
        output['pre_learning'] = np.mean(pre_test_error ** 2)  # errors
        output['post_learning'] = np.mean(test_error ** 2)  # errors
        output['pre_learning_test'] = np.mean(pre_test2_error ** 2)  # errors
        output['post_learning_test'] = np.mean(test2_error ** 2)  # errors
        output['RMSE'] = np.sqrt(output["post_learning_test"])
        output['presentation_number'] = kwargs['presentation_number']
        output['hidden_layer'] = kwargs['hidden_layer']
        output['learning_rate'] = kwargs['learning_rate']
        output['momentum'] = kwargs['momentum']

        output['ind_learning'] = kwargs['ind_learning']
        output['ind_testing'] = kwargs['ind_testing']
        for i in range(len(kwargs["test_dataset"]['x'])):
            output['ind{i}'.format(i=i)] = test_output2[i]

        learn_index = (output['pre_learning'] - output['post_learning']) / output['post_learning']

        test_index = (output['pre_learning_test'] - output['post_learning_test']) / output[
            'post_learning_test']

        output['index_learn'] = 100 * learn_index
        output['index_test'] = 100 * test_index
        output['selected_var'] = kwargs['selected_var']

        kwargs.pop('dataset')

        return output


def combination_var(id_job, n_workers, kwargs_list):

    output = 'combinations_{}'.format(id_job)

    supervisor = Supervisor(n_workers=n_workers,
                            output_file=output,
                            back_up_frequency=5000,
                            kwargs_list=kwargs_list,
                            id_job=id_job)

    supervisor.launch_test()


if __name__ == '__main__':

    print("Call a function")
