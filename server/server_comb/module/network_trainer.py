# -*- coding: utf8 -*-

import numpy as np
from module.mlp import MLP


class NetworkTrainer(object):

    def __init__(self):

        self.network = None

    def create_network(self, dataset, hidden_layer):

        input_number = dataset['x'].shape[1]
        if len(dataset['y'].shape) > 1:
            output_number = dataset['y'].shape[1]
        else:
            output_number = 1

        # Create the network

        args = tuple()
        args += input_number,
        for i in range(len(hidden_layer)):
            args += hidden_layer[i],
        args += output_number,

        self.network = MLP(*args)

    def test_the_network(self, dataset):

        n = dataset['x'].shape[0]

        if len(dataset['y'].shape) > 1:
            errors = np.zeros((n, dataset['y'].shape[1]))
            output = np.zeros((n, dataset['y'].shape[1]))
        else:
            errors = np.zeros(n)
            output = np.zeros(n)

        for i in range(n):
            output[i] = self.network.propagate_forward(dataset['x'][i])
            expected_output = np.asarray(dataset['y'][i])
            errors[i] = expected_output - output[i]

        return errors, output

    def teach_the_network(self, presentation_number, dataset, learning_rate=0.1, momentum=0.1):

        # Create order with index that will be randomized
        order = np.arange(dataset["x"].shape[0])

        # Each presentation, the perceptron receives all the inputs and outputs of the dataset (but in
        # a different order)

        for i in range(presentation_number):

            # Each presentation, use a different order
            np.random.shuffle(order)

            for j in order:
                self.network.propagate_forward(dataset['x'][j])

                expected_output = np.asarray(dataset['y'][j])
                self.network.propagate_backward(target=expected_output,
                                                lrate=learning_rate,
                                                momentum=momentum)

    def reset_the_network(self):

        self.network.reset()