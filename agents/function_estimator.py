from mdp.states import state_action_reducer
from nn.neuralnet import NeuralNet
import numpy as np

class FunctionEstimator:
    def get_qval(self, state, action):
        raise NotImplementedError

    def get_grad(self, state, action):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError
    
    def get_weights(self):
        raise NotImplementedError

class NeuralNetEstimator(FunctionEstimator):
    def __init__(self, sample_state, sample_action, hidden_layer_sizes):
        sample_input = state_action_reducer(sample_state, sample_action)
        nn_sizes = [sample_input.shape[1]] + hidden_layer_sizes + [1]

        self.nn = NeuralNet(nn_sizes, 0.9) #Alpha parameter is ignored
    
    def get_qval(self, state, action):
        input_form = state_action_reducer(state, action)
        return self.nn.predict(input_form)[0, 0]
    
    def get_grad(self, state, action):
        input_form = state_action_reducer(state, action)
        return self.nn.gradient((input_form, np.mat([0])))
    
    def set_weights(self, weights):
        self.nn.set_weights(weights)
    
    def get_weights(self):
        return self.nn.get_weights()

