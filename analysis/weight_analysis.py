import numpy as np
import matplotlib.pyplot as plt
from module.save import Database


class ExtractData:

    def __init__(self, filename, cutoff):

        self.filename = filename
        self.db = Database(self.filename)

        self.cutoff = cutoff

    def cleaned_networks(self):

        test_index = self.db.read_column(column_name='index_test')
        test_index = [float(i) for i in test_index]

        test_value = self.db.read_column(column_name='post_learning_test')
        test_value = [float(i) for i in test_value]

        good_networks = []

        for i in range(len(test_index)):
            if test_index[i] > self.cutoff and test_value[i] < 0.07:
                good_networks.append(i)

        return good_networks


class Supervisor:

    def __init__(self, filename, cutoff):

        self.cutoff = cutoff
        self.extract = ExtractData(filename=filename, cutoff=self.cutoff)

    def compute_weights(self):

        print("*********************************")

        networks = self.extract.cleaned_networks()

        print("Number of selected networks : {}".format(len(networks)))
        print("Cutoff was : {}".format(self.cutoff))
        print("*********************************")

        results_init = np.zeros((53, 52))
        results_test = np.zeros((53, 52))
        results_init_2 = np.zeros((52, 3))
        results_test_2 = np.zeros((52, 3))

        for i in networks:
            data_init = np.loadtxt('weights/weights_init_{}.txt'.format(i))
            results_init = data_init

            data_test = np.loadtxt('weights/weights_test_{}.txt'.format(i))
            results_test += data_test

            data_init_2 = np.loadtxt('weights/weights_init_2_{}.txt'.format(i))
            results_init_2 = data_init_2

            data_test_2 = np.loadtxt('weights/weights_test_2_{}.txt'.format(i))
            results_test_2 += data_test_2

        return results_init, results_test, results_init_2, results_test_2

    def make_graph(self):

        results_init, results_test, results_init_2, results_test_2 = self.compute_weights()

        a = np.zeros((53,52))

        c = []

        for i, j in enumerate(results_test):
            b = np.sort(np.absolute(j), )
            a[i] = b

            c.append(np.mean(np.absolute(j)))

        d = c.copy()
        c.sort()
        print(c)
        print([np.where(c == d[i])[0][0] for i in range(len(d))])

        plt.imshow(a, interpolation='nearest', cmap='hot')
        plt.colorbar()
        plt.show()


        # plt.subplot(121)
        # plt.imshow(results_test, interpolation='nearest', cmap='RdBu', vmin=-5, vmax=5)
        # plt.title('Input-hidden layer weights after learning')
        # plt.colorbar()
        # plt.subplot(122)
        # plt.imshow(results_test_2, interpolation='nearest', cmap='RdBu', vmin=-1.7, vmax=1.7)
        # plt.title('Hidden_layer-output weights after learning')
        # plt.colorbar()
        # plt.xticks([0,1,2])
        # plt.show()


        # plt.subplot(121)
        # plt.imshow(results_init, interpolation='nearest', cmap='RdBu')
        # plt.title('First layer: initialization stage')
        # plt.colorbar()
        # plt.subplot(122)
        # plt.title('First layer: post-learning')
        # plt.imshow(results_test, interpolation="nearest", cmap='RdBu')
        # plt.colorbar()
        # plt.savefig('graphs/layer1.png', dpi=300)
        # plt.show()
        #
        # plt.subplot(121)
        # plt.title('Second layer: initialization stage')
        # plt.imshow(results_init_2, interpolation="nearest", cmap='RdBu')
        # plt.xticks([0,1,2])
        # plt.colorbar()
        # plt.subplot(122)
        # plt.title('Second layer: post-learning')
        # plt.imshow(results_test_2, interpolation="nearest", cmap='RdBu')
        # plt.xticks([0,1,2])
        # plt.colorbar()
        # plt.savefig('graphs/layer2.png', bbox_inches = 'tight', dpi=300)
        # plt.show()

if __name__ == "__main__":

    supervisor = Supervisor(filename='results_all_output', cutoff=500)
    supervisor.make_graph()