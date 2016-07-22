import numpy as np
from itertools import combinations


class DataManager(object):
    def __init__(self, file_name="dataset_290416_3output", explanans_size=52, explanandum_size=3):

        self.folder_path = "../../data"
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

