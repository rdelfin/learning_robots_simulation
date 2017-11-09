from unittest import main, TestCase
from nn.neuralnet import NeuralNet

import numpy as np

delta = 0.000000000001

class NeuralNetTest(TestCase):
    def test_cost(self):
        training_x = np.mat([[1, 2], [3, 4]], dtype=np.float64)
        training_y = np.mat([[2], [2]], dtype=np.float64)
        training_examples = (training_x, training_y)
        weights = [
            # W^(0)
            0.3,  0.01,
            -0.2, 0.2,
            0.1,  -0.5,
            # W^(1)
            0.3,
            0.2,
            0.05
        ]
        test_nn = NeuralNet([2, 2, 1], 0.1)
        test_nn.set_weights(weights)

        #hidden_1 = [0.5, 0.549833997]   # Mid-point calculations
        #hidden_2 = [0.477515175, 0.581759377] # Mid-point calculations
        #output = np.mat([[0.295503035], [0.331302075]]) #Mid-point calculations
        expected_cost = 1.422465667

        self.assertAlmostEqual(expected_cost, test_nn.cost(training_examples), places=5)
        

    def test_gradient(self):
        training_x = np.mat([[1, 2, 5, 4], [5, 2, 7, 9]], dtype=np.float64)
        training_y = np.mat([[1, 2, 1], [5, 4, 3]], dtype=np.float64)
        training_examples = (training_x, training_y)
        test_nn = NeuralNet([4, 7, 5, 3], 0.1)
        weights = test_nn.get_weights()

        # Get cost according to different weights
        estimate_grad = np.zeros_like(weights)

        for i in range(len(estimate_grad)):
            del_vector = np.eye(1, len(weights), i) * delta / 2
            weights_lower = np.array(weights - del_vector)
            weights_upper = np.array(weights + del_vector)
            test_nn.set_weights(weights_lower)
            lower_cost = test_nn.cost(training_examples)
            test_nn.set_weights(weights_upper)
            upper_cost = test_nn.cost(training_examples)

            estimate_grad[i] = (upper_cost - lower_cost) / delta

        test_nn.set_weights(weights)
        backprop_grad = test_nn.gradient(training_examples)

        self.assertLessEqual(max(np.abs(backprop_grad - estimate_grad)), max(backprop_grad) * 0.05)


if __name__ == "__main__": # pragma: no cover
    main()
