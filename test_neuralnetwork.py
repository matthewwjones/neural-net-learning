import os
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from neuralnet.neuralnetwork import NeuralNetwork


class TestNeuralNetwork(TestCase):

    def test_query(self):
        input_nodes = 3
        output_nodes = 3
        hidden_nodes = 3
        learning_rate = 0.3
        neuralnetwork = NeuralNetwork(input_nodes, output_nodes, hidden_nodes, learning_rate)

        training_data = os.path.join(os.path.dirname(__file__), 'mnist_train_100.csv')

        with open(training_data, 'r') as data_file:
            data_list = data_file.readlines()

        all_values = data_list[0].split(',')
        image_array = np.asarray(all_values[1:], 'float', ).reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        # plt.show()
        self.assertEqual(len(image_array[0]), 28)
