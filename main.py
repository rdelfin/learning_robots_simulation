import simulator
from simulator import Simulation
from mdp.states import FullState, Action
from agents.function_estimator import NeuralNetEstimator
from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent
from firebase_scripts import travel_data

import matplotlib.pyplot as plt
import numpy as np

ep_count = 1000

def get_nn_sarsa_agent(eps, alpha, gamma, layers):
    sample_state = simulator.get_initial_state(travel_data.get_locations())
    sample_action = Action(True, '77d0b6d7-2a69-4d81-9e29-1bac4e5f5c94')
    estimator = NeuralNetEstimator(sample_state, sample_action, layers)
    return SarsaAgent(eps, alpha, gamma, estimator)

if __name__ == "__main__":
    sarsa_agent = get_nn_sarsa_agent(0.1, 0.5, 1, [20, 10])
    random_agent = RandomAgent()

    sarsa_sim = Simulation(sarsa_agent, 1)
    random_sim = Simulation(random_agent, 1)

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
        print("\tSARSA episode...")
        sarsa_vals += [sarsa_sim.run_episode()]
        print("\tDone")
        print("\tRandom episode...")
        random_vals += [random_sim.run_episode()]
        print("\tDone")
        print()

    print("Average SARSA reward: %.5f ± %.2f" % (np.mean(sarsa_vals), np.std(sarsa_vals, ddof=1)))
    print("Average Random reward: %.5f ± %.2f" % (np.mean(random_vals), np.std(random_vals, ddof=1)))
    print(sarsa_vals)
    print(random_vals)
    print()
    print("Plots:")

    f, axarr = plt.subplots(1, 2)
    axarr[0].plot(range(ep_count), sarsa_vals)
    axarr[0].set_title('SARSA Agent')
    axarr[1].plot(range(ep_count), random_vals)
    axarr[1].set_title('Random Agent')

    plt.show()
