from unittest import main, TestCase
from nn.neuralnet import NeuralNet

import numpy as np

delta = 0.01

class NeuralNetTest(TestCase):
    def test_gradient(self):
        training_x = np.mat([[1, 2, 5, 4], [5, 2, 7, 9]])
        training_y = np.mat([[1, 2, 1], [5, 4, 3]])
        training_examples = (training_x, training_y)
        test_nn = NeuralNet([4, 7, 5, 3], 0.1)
        weights = test_nn.get_weights()
        weights_lower = weights - (np.ones(len(weights)) * delta / 2)
        weights_upper = weights + (np.ones(len(weights)) * delta / 2)

        # Get cost according to different weights
        test_nn.set_weights(weights_lower)
        lower_cost = test_nn.cost(training_examples)
        test_nn.set_weights(weights_upper)
        upper_cost = test_nn.cost(training_examples)

        backprop_grad = test_nn.gradient(training_examples)
        estimate_grad = (upper_cost - lower_cost) / delta

        print("BACK PROP GRADIENT: ")
        print(backprop_grad)
        print("\nESTIMATE GRADIENT: ")
        print(estimate_grad)
        self.assertLessEqual(max(np.abs(backprop_grad - estimate_grad)), max(backprop_grad) * 0.05)


if __name__ == "__main__": # pragma: no cover
    main()
