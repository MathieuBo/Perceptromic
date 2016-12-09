import numpy as np
from itertools import combinations
from tqdm import tqdm
import pickle


class DataManager(object):
    def __init__(self, file_name="dataset_020916", explanans_size=130, explanandum_size=3, add_random=True):

        self.folder_path = "data"
        self.file_path = "{}/{}.txt".format(self.folder_path, file_name)
        self.explanans_size = explanans_size
        self.explanandum_size = explanandum_size
        self.data = self.import_txt(add_random=add_random)

    def import_txt(self, add_random):

        print("Import txt file.")

        data = np.loadtxt(self.file_path)

        # Add X random columns to analyze performance of random data
        if add_random is True:

            for i in np.arange(23):
                to_insert = np.random.random(data.shape[0])
                data = np.insert(data, -3, to_insert, axis=1)
        else:
            pass

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

    @staticmethod
    def format_random(data):

        mean, std = np.mean(data), np.std(data)
        if std != 0:
            formatted_data = (data - mean) / std
        else:
            formatted_data = (data - mean)

        return formatted_data


def job_definition(total_comb):

    n_job = 100 * (total_comb // 100000)

    step = total_comb // n_job

    print('\ntotal number of jobs: {}'.format(n_job))

    # We know that 4hours (240 minutes) for 100,000 networks is ok
    # So we choose the number of jobs according to that

    mean_time = step * 240 / 2000

    print("mean job time should be : {}h{}\n".format(int(mean_time // 60), int(mean_time % 60)))

    args = list()

    for i in range(n_job):
        if i != n_job-1:
            start = i + i * step
            end = start + step + 1

            args.append((start, end))
        else:
            start = i + i * step
            end = start + step + total_comb % step + 1 - i
            args.append((start, end))

    return args


def prepare_comb_list(explanans_size=130, combination_size=3):

    comb_list = [i for i in combinations(np.arange(explanans_size), combination_size)]

    np.save("comb_list.npy", comb_list)

    return comb_list


class Supervisor:

    def __init__(self, add_random, explanans_size, combination_size):

        self.n_network = 50

        self.combination_list = prepare_comb_list(explanans_size=explanans_size, combination_size=combination_size)

        # Import txt file
        self.data_manager = DataManager(add_random=add_random,explanans_size=explanans_size, explanandum_size=3)
        self.data_manager.format_data()  # Center-reduce input variables and normalize output variables

        self.indexes_list = SamplesCreator.combinations_samples(n=self.data_manager.data.shape[0],
                                                                split_value=int(0.8 * self.data_manager.data.shape[0]))

    @staticmethod
    def convert_seconds_to_h_m_s(seconds):

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def prepare_kwargs_list(self, id_job, start, end):

        n_network = 50
        hidden_layer = 3

        learning_rate = 0.05
        presentation_number = 1000

        kwargs_list = []

        for selected_variables in self.combination_list[start:end]:

            for selected_ind in self.indexes_list[0:n_network]:

                samples_learning = self.data_manager.import_data(explanandum=[0, 1, 2],
                                                                 explanans=selected_variables,
                                                                 individuals=selected_ind['learning'])

                samples_testing = self.data_manager.import_data(explanandum=[0, 1, 2],
                                                                explanans=selected_variables,
                                                                individuals=selected_ind['testing'])

                kwargs = {"dataset": samples_learning,
                          "test_dataset": samples_testing,
                          "hidden_layer": [hidden_layer],
                          "presentation_number": presentation_number,
                          "learning_rate": learning_rate,
                          "momentum": learning_rate,
                          'ind_learning': selected_ind['learning'],
                          'ind_testing': selected_ind['testing'],
                          'selected_var': selected_variables
                          }

                kwargs_list.append(kwargs)

        filename = "perceptromic_job{}".format(id_job)

        path_to_file = "kwargs/{}.p".format(filename)

        pickle.dump(kwargs_list, open(path_to_file, "wb"))


def combination_var():

    print("\n*************************")
    print('Preparing kwarg list...')
    print("**************************\n")

    s = Supervisor(add_random=True, explanans_size=130, combination_size=3)

    print("\n")

    list_jobs = job_definition(total_comb=len(s.combination_list))

    for id_job, job in tqdm(enumerate(list_jobs)):

        s.prepare_kwargs_list(id_job=id_job, start=job[0], end=job[1])

    print("\n**************************")
    print('Kwarg list ready.')
    print("\n*************************")


if __name__ == '__main__':

    combination_var()
