import numpy as np
import scipy.special

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.normal(0.0, pow(self.input_nodes, - 0.5),
                                                     (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = np.random.normal(0.0, pow(self.hidden_nodes, - 0.5),
                                                      (self.output_nodes, self.hidden_nodes))
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # convert inputs into 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)

        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        # hidden layer errors = output errors, split by weights, recombined at hidden nodes.
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.weights_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                                  np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.weights_input_hidden += self.learning_rate * np.dot(
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)),
            np.transpose(inputs))

    def query(self, inputs_list):
        # convert inputs to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)

        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
