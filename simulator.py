from firebase_scripts import travel_data
from mdp.states import FullState, Request, Action, is_terminal_state, transition_func, reward_func
from mdp.mdp import MDP
import random

class Simulation:
    def __init__(self, agent, discount_factor):
        self.discount_factor = discount_factor
        self.locations = travel_data.get_locations()
        self.location_ids = [key for key in self.locations]
        self.mdp = MDP(transition_func, reward_func, self.discount_factor, is_terminal_state)
        self.agent = agent

    def generate_initial_state(self):
        self.initial_state = get_initial_state(self.locations)

    def run_episode(self):
        self.generate_initial_state()
        self.mdp.reset_world(self.initial_state)

        while not self.mdp.is_terminal():
            state = self.mdp.current_state()
            new_reward_state = self.mdp.step(self.agent.next_action(state))
            self.agent.action_update(*new_reward_state)

        return self.mdp.get_episode_utility()

def get_initial_state(locations):
    return FullState(location=random.choice(list(locations.keys())),
                     time_of_day=32400, # 9:00 am, number of seconds
                     day_of_week=random.randint(0, 6),   # Monday: 0, ..., Sunday: 6
                     person_present_map={person: None for person in locations},
                     request_history=[])
