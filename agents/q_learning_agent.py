from agents.agent import Agent
from nn.neuralnet import NeuralNet
from firebase_scripts import travel_data

from functools import reduce
import random

import numpy as np

class QLearningAgent(Agent):
    def __init__(self, eps):
        # FIX: Stop depending on states class to define sizes
        # 8 input neurons:
        #   1 for time
        #   7 for each day of week
        #   num_actions for each action (move to location, excluding the current location)
        self.travel_locations = travel_data.get_location_ids()
        self.num_actions = len(self.travel_locations) - 1
        self.neural_net = NeuralNet([8 + self.num_actions, 20, 10, 1], 0.9)
        self.eps = eps

    def next_action(self, state):
        max_location = max(self.travel_locations, key=lambda loc: self.neural_net.predict(self.to_nn_input(loc, state))[0, 0])
        rand_location = random.choice(self.travel_locations)

        self.action_taken = rand_location if random.random() < self.eps else max_location   # Epsilon-greedy behaviour
        return self.action_taken
    
    def to_nn_input(self, location, state):
        nn_input = np.array([0.0 for x in range(8 + self.num_actions)], np.float64)
        func_state = state.time_only_state()
        travel_id_pairs = zip(self.travel_locations, range(len(self.travel_locations)))
        location_idx = reduce(lambda x, y: y[1] if y[0] == location else x, travel_id_pairs, -1)
        if location_idx < 0: # Should never happen
            raise IndexError

        nn_input[0] = func_state[0]            # Add time of day to state.
        nn_input[1 + func_state[1]] = 1.0      # set to 1 the neuron corresponding to the current day (skipping the first element).
        nn_input[8 + location_idx] = 1.0       # Set 1 to the neuron corresponding to the action taken.

        return np.mat(nn_input)


    def action_update(self, reward, new_state):
        # TODO: Implement update (page 198 of Sutton http://incompleteideas.net/sutton/book/bookdraft2017nov5.pdf)
        pass