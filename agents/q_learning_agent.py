from agents.agent import Agent
from nn.neuralnet import NeuralNet
from firebase_scripts import travel_data
from agents.function_estimator import FunctionEstimator, NeuralNetEstimator
from mdp.states import Action, FullState

from functools import reduce
import random

import numpy as np

class QLearningAgent(Agent):
    def __init__(self, eps, estimator: FunctionEstimator):
        # FIX: Stop depending on states class to define sizes
        # 8 input neurons:
        #   1 for time
        #   7 for each day of week
        #   num_actions for each action (move to location, excluding the current location)
        self.travel_locations = travel_data.get_location_ids()
        self.num_actions = len(self.travel_locations) - 1
        self.neural_net = NeuralNet([8 + self.num_actions, 20, 10, 1], 0.9)
        self.eps = eps
        self.estimator = estimator

    def next_action(self, state: FullState):
        max_location = max(self.travel_locations, key=lambda loc: self.estimator.get_qval(state, Action(loc, True)))
        rand_location = random.choice(self.travel_locations)

        self.action_taken = rand_location if random.random() < self.eps else max_location   # Epsilon-greedy behaviour
        return self.action_taken

    def action_update(self, reward, new_state: FullState):
        # TODO: Implement update (page 198 of Sutton http://incompleteideas.net/sutton/book/bookdraft2017nov5.pdf)
        pass