from unittest import main, TestCase
from nn.neuralnet import NeuralNet

import numpy as np

delta = 0.1

class NeuralNetTest(TestCase):
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
            weights_lower = weights - del_vector
            weights_upper = weights + del_vector

            print("WEIGHTS UPPER: " + str(weights_upper))
            print("WEIGHTS LOWER: " + str(weights_lower))


            test_nn.set_weights(weights_lower)
            lower_cost = test_nn.cost(training_examples)
            test_nn.set_weights(weights_upper)
            upper_cost = test_nn.cost(training_examples)
            estimate_grad[i] = (upper_cost - lower_cost) / delta

        test_nn.set_weights(weights)
        backprop_grad = test_nn.gradient(training_examples)

        print("BACK PROP GRADIENT: ")
        print(backprop_grad)
        print("\nESTIMATE GRADIENT: ")
        print(estimate_grad)
        self.assertLessEqual(max(np.abs(backprop_grad - estimate_grad)), max(backprop_grad) * 0.05)


if __name__ == "__main__": # pragma: no cover
    main()
