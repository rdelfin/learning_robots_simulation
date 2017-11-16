from firebase_scripts import travel_data, schedule_data
from mdp.states import FullState
from mdp.mdp import MDP
import random

def transition_func(state, action):
    pass

def reward_func(state, action, newState):
    if(len(state.request_history) > len(newState.request_history)):
        latest_request = state.request_history[-1]
        if(state.person_present_map[latest_request.location]):
            loc_schedule_data = schedule_data.get_schedule_data()[latest_request.location]
            p_answer = loc_schedule_data["panswer"] if "panswer" in loc_schedule_data else 1
            return 1 if random.random() < p_answer else 0

    
    return 0

class Simulation:
    def __init__(self):
        self.discount_factor = 0.9
        self.locations = travel_data.get_locations()
        self.location_ids = [key for key in self.locations]
        initial_state = FullState(location=random.choice(self.locations),
                                  time_of_day=540, # 9:00 am, number of minutes
                                  day_of_week=random.randint(0, 6),   # Monday: 0, ..., Sunday: 6
                                  person_present_map={person: False for person in self.locations},
                                  request_history=[])
        self.mdp = MDP(transition_func, reward_func, self.discount_factor)

    def run_episode(self):
        pass
