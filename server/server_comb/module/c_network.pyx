# -*- coding: utf8 -*-

import numpy as np
cimport numpy as cnp


cdef class MLP:

    """ Multi-layer perceptron class. """

    cdef:
        tuple shape
        list layers, weights, dw
        int n


    def __init__(self, *args):

        cdef:
            int i

        """ Initialization of the perceptron with given sizes.  """

        self.shape = args

        self.n = len(args)

        # Build layers

        self.layers = []

        # Input layer (+1 unit for bias)

        self.layers.append(np.ones(self.shape[0]+1))

        # Hidden layer(s) + output layer

        for i in range(1, self.n):

            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)

        self.weights = []

        for i in range(self.n-1):

            self.weights.append(np.zeros((self.layers[i].size,

                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)

        self.dw = [0,] * len(self.weights)

        # Reset weights

        self.reset()

    cpdef reset(self):

        cdef:
            int i
            cnp.ndarray Z

        """ Reset weights """

        for i in range(len(self.weights)):

            Z = np.random.random((self.layers[i].size, self.layers[i+1].size))

            self.weights[i][...] = (2*Z-1)*0.25

    cpdef cnp.ndarray propagate_forward(self, cnp.ndarray data):

        cdef:
            int i

        """ Propagate data from input layer to output layer. """

        # Set input layer

        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function

        for i in range(1, self.n):

            # Propagate activity

            self.layers[i][...] = self.sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))

        # Return output

        return self.layers[-1]

    cpdef propagate_backward(self, cnp.ndarray target, float lrate=0.001, float momentum=0.2):

        cdef:
            list deltas
            cnp.ndarray error, delta, layer, dw
            int i


        """ Back propagate error related to target using lrate. """

        deltas = []

        # Compute error on output layer

        error = target - self.layers[-1]

        delta = error * self.dsigmoid(self.layers[-1])

        deltas.append(delta)

        # Compute error on hidden layers

        for i in range(self.n-2, 0, -1):

            delta = np.dot(deltas[0], self.weights[i].T) * self.dsigmoid(self.layers[i])

            deltas.insert(0, delta)

        # Update weights

        for i in range(self.n-1):

            layer = np.atleast_2d(self.layers[i])

            delta = np.atleast_2d(deltas[i])

            dw = np.dot(layer.T, delta)

            self.weights[i] += lrate*dw + momentum*self.dw[i]

            self.dw[i] = dw

    cdef cnp.ndarray return_error(self, cnp.ndarray target):

        cdef:
            cnp.ndarray error

        error = np.zeros(len(self.layers[-1]))
        error[:] = target - self.layers[-1]

        return error


    cdef cnp.ndarray sigmoid(self, cnp.ndarray x):

        """ Sigmoid like function using tanh """
        return np.tanh(x)

    cdef cnp.ndarray dsigmoid(self, cnp.ndarray x):

        """ Derivative of sigmoid above """

        return 1.0 - x ** 2
