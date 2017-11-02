import numpy as np

class NeuralNet:
    """
    Represents a feed forward, fully connected neural network that can be trained using
    stochastic gradient descent. This one makes use of the sigmoid activation function
    for all but the last neuron, which uses a linear activation.
    """
    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.layers = len(sizes)
        self.reset_weights()

    def reset_weights(self):
        """
        Resets all the weights in the neural network to a normal distribution around
        0 with standard deviation of 0.1.
        """
        self.weights = []
        for i in range(self.layers - 1):
            self.weights = np.append(self.weights,
                                     np.random.normal(scale=0.1, size=tuple(self.sizes[i:i+2])))

    def get_layer_weights(self, layer):
        """
        Gets the matrix of weights between a given layer and the next layer from the
        array self.weights. The dimensions are set up so that you can calculate the
        next layer's activation as:
        ```
        activation(get_layer_weights(layer) * last_layer_activation_vector)
        ```
        Where last_layer_activation_vector is a column (aka vertical) vector.
        """
        dims = (self.sizes[layer+1], self.sizes[layer])
        size = dims[0] * dims[1]
        next_pairs = ((self.sizes[i], self.sizes[i+1]) for i in self.layers - 1)
        offset = sum(pair[0] * pair[1] for pair in next_pairs)

        weight_subset = self.weights[offset:offset+size]
        weight_subset.resize(dims)

        return np.matrix(weight_subset)

    def add_sample(self, sample_x, sample_y):
        pass

    def train(self):
        pass

    def get_weights(self):
        pass

    def predict(self, input_x):
        pass

    def cost(self):
        pass

    def accuracy(self, test_set):
        pass
