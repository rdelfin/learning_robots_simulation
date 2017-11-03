import numpy as np

def activation(x):
    """
    Returns the sigmoid activation function for all values in x
    """
    return 1.0 / 1 + np.exp(-x)

def d_activation(x):
    """
    Implements the derivative of the sigmoid activation function
    for all values of x
    """
    a = activation(x)
    return a * (1 - a)

class NeuralNet:
    """
    Represents a feed forward, fully connected neural network that can be trained using
    stochastic gradient descent. This one makes use of the sigmoid activation function
    for all but the last neuron, which uses a linear activation.
    """
    def __init__(self, sizes, alpha):
        self.sizes = list(sizes)
        self.layers = len(sizes)
        self.samples = np.array([])
        self.alpha = alpha
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


    def set_weights(self, weights):
        """
        Set the weights for the neural net. It is assumed that you pass in a numpy
        array of the right size
        """
        self.weights = weights.copy()


    def get_layer_weights(self, layer):
        """
        Gets the matrix of weights between a given layer and the next layer from the
        array self.weights. The dimensions are set up so that you can calculate the
        next layer's activation as:
        ```
        last_layer_activation_vector * activation(get_layer_weights(layer))
        ```
        Where last_layer_activation_vector is a row (aka horizontal) vector.
        """
        dims = (self.sizes[layer], self.sizes[layer+1])
        size = dims[0] * dims[1]
        next_pairs = ((self.sizes[i], self.sizes[i+1]) for i in range(layer))
        offset = sum(pair[0] * pair[1] for pair in next_pairs)

        weight_subset = self.weights[offset:offset+size].copy()
        weight_subset.resize(dims)

        return np.matrix(weight_subset)

    def add_sample(self, sample_x, sample_y):
        """
        Add a sample for training in the next call of the train() method
        """
        if(len(self.samples)):
            self.samples = np.array([[sample_x, sample_y]])
        else:
            self.samples = np.concatenate(self.samples, [[sample_x, sample_y]])

    def train(self):
        """
        Trains the neural network with all the samples added with the
        add_sample() method since the last time the network was trained.
        """
        training_examples = self.samples.copy()
        self.samples = np.array([])

        grad_vec = self.gradient(training_examples)
        self.set_weights(self.weights - self.alpha*grad_vec)


    def get_weights(self):
        """
        Returns all the weights as a flat vector. It is guaranteed to be a copy of
        the weights used internally.
        """
        return self.weights.copy()

    def predict(self, input_x):
        """
        Takes in a given sample (input_x) and calculates the predicted values of
        the output nodes on the neural network.
        """
        a_last = input_x
        for layer in range(self.layers - 1):
            z = a_last * self.get_layer_weights(layer)
            a_last = activation(z) if layer != self.layers - 2 else z

        return a_last

    def cost(self, training_set):
        pass

    def accuracy(self, test_set):
        pass

    def gradient(self, training_set):
        """
        Calculates the gradient vector for the neural network of the cost
        function over the training set with respect to the weights.
        """
        pass
