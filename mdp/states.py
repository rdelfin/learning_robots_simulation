import numpy as np
from firebase_scripts import travel_data
from functools import reduce

class Request:
    def __init__(self, *, location, time_of_day, day_of_week):
        self.location = location
        self.time_of_day = time_of_day
        self.day_of_week = day_of_week
    
    def __repr__(self):
        return "Request(location=%s, time_of_day=%d, day_of_week=%d)" % \
                (self.location, self.time_of_day, self.day_of_week)

class Action:
    def __init__(self, travel=False, location=None):
        self.travel = travel
        self.location = location
    
    def __repr__(self):
        return "Action(travel=%s, location=%s)" % (str(self.travel), self.location)

def is_terminal_state(state):
    return state.time_of_day > 64800  # 6pm in seconds

class FullState:
    def __init__(self, *, location, time_of_day, day_of_week, person_present_map, request_history):
        self.location = location
        self.time_of_day = time_of_day
        self.day_of_week = day_of_week
        self.person_present_map = person_present_map
        self.request_history = request_history

    def time_only_state(self):
        return (self.time_of_day, self.day_of_week)

    def time_and_location_state(self):
        return (self.time_of_day, self.day_of_week, self.location)

    def __repr__(self):
        return "FullState(location=%s, time_of_day=%d, day_of_week=%d," \
                "person_present_map=%s)" % \
                (self.location, self.time_of_day, self.day_of_week, str(self.person_present_map))

def state_action_reducer(state, action):
    travel_locations = travel_data.get_location_ids()
    num_actions = len(travel_locations)

    vec_input = np.array([0.0 for x in range(8 + num_actions)], np.float64)
    func_state = state.time_only_state()
    travel_id_pairs = zip(travel_locations, range(len(travel_locations)))
    location_idx = reduce(lambda x, y: y[1] if y[0] == action.location else x, travel_id_pairs, -1)
    if location_idx < 0: # Should never happen
        raise IndexError

    vec_input[0] = func_state[0]            # Add time of day to state.
    vec_input[1 + func_state[1]] = 1.0      # set to 1 the element corresponding to the current day (skipping the first element).
    vec_input[8 + location_idx] = 1.0       # Set 1 to the element corresponding to the action taken.

    return np.mat(vec_input)