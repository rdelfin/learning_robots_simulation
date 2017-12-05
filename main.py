#!/usr/bin/env python3
import sys
import argparse

import simulator
from simulator import Simulation
from mdp.states import FullState, Action
from agents.function_estimator import NeuralNetEstimator
from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent
from firebase_scripts import travel_data
import numpy as np

ep_count = 200

def get_nn_sarsa_agent(eps, alpha, gamma, layers):
    sample_state = simulator.get_initial_state(travel_data.get_locations())
    sample_action = Action(True, '77d0b6d7-2a69-4d81-9e29-1bac4e5f5c94')
    estimator = NeuralNetEstimator(sample_state, sample_action, layers)
    return SarsaAgent(eps, alpha, gamma, estimator)

def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epsilon",  type=float, help="Epsilon for epsilon-greedy algorithm")
    parser.add_argument("-a", "--alpha",    type=float, help="Learning rate for approximate SARSA algorithm")
    parser.add_argument("-d", "--discount", type=float, help="Discount factor for the MDP")
    parser.add_argument("-l", "--layers",    type=int,nargs='+', help="Number of neurons at each layer of the neural network")

    args = parser.parse_args()

    eps = 0.1
    alpha = 0.5
    discount = 1.0
    layers = [20, 10]

    if args.epsilon:
        eps = args.epsilon
    if args.alpha:
        alpha = args.alpha
    if args.discount:
        discount = args.discount
    if args.layers:
        layers = args.layers

    return (eps, alpha, discount, layers)

# Main Code:
#eps, alpha, discount, layers = parse_my_args()
if __name__ == "__main__":
    eps = float(input("Epsilon: "))
    alpha = float(input("Alpha: "))
    discount = float(input("Discount: "))
    layers = [int(x) for x in input("Layers: ").split()]
    print()
    print("Configuration:")
    print("\tEpsilon:  %f" % eps)
    print("\tAlpha:    %f" % alpha)
    print("\tDiscount: %f" % discount)
    print("\tLayers:   %s" % str(layers))

    sarsa_agent = get_nn_sarsa_agent(eps, alpha, discount, layers)
    random_agent = RandomAgent()

    sarsa_sim = Simulation(sarsa_agent, discount)
    random_sim = Simulation(random_agent, discount)

    sarsa_vals = []
    random_vals = []

    print("====================================")
    print("          AGENT COMPARISON          ")
    print("====================================")

    print()
    print("SARSA agent with NN vs Random Action agent")
    print()
    print("Simulating...")
    for i in range(1, ep_count+1):
        print("Episode %d" % i)
        print("\tSARSA episode...", end="")
        sys.stdout.flush()
        sarsa_vals += [sarsa_sim.run_episode()]
        print(" Done. Reward: %d" % sarsa_vals[-1])
        print("\tRandom episode...", end="")
        sys.stdout.flush()
        random_vals += [random_sim.run_episode()]
        print(" Done. Reward: %d" % random_vals[-1])
        print()
    
    print("Average SARSA reward: %.5f ± %.2f" % (np.mean(sarsa_vals), np.std(sarsa_vals, ddof=1)))
    print("Average Random reward: %.5f ± %.2f" % (np.mean(random_vals), np.std(random_vals, ddof=1)))
    print()
    print("SARSA points: %s" % str(sarsa_vals))
    print("Random points: %s" % str(random_vals))
    print()
    #print()
    #print("Plots:")

    #f, axarr = plt.subplots(1, 2)
    #axarr[0].plot(range(ep_count), sarsa_vals)
    #axarr[0].set_title('SARSA Agent')
    #axarr[1].plot(range(ep_count), random_vals)
    #axarr[1].set_title('Random Agent')

    #plt.show()
