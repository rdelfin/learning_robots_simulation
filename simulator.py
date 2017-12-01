from firebase_scripts import travel_data, schedule_data
from mdp.states import FullState, Request, Action, is_terminal_state
from mdp.mdp import MDP
import random
import util
import scipy
import scipy.stats

# Constants: replace with values in person tables
arrival_stddev = 120
departure_stddev = 180
open_stddev = 300

def probability_interval(interval, start_time, end_time, stddev, arrival):
    mean = interval.start.second if arrival else interval.end.second
    distribution = scipy.stats.norm(mean, stddev)
    return distribution.cdf(end_time) - distribution.cdf(start_time)

def sample_probability_interval(interval, start_time, end_time, stddev, arrival):
    mean = interval.start.second if arrival else interval.end.second
    min_trunc, max_trunc = (start_time - mean) / stddev, (end_time - mean) / stddev
    return scipy.stats.truncnorm.rvs(min_trunc, max_trunc, loc=mean, scale=stddev)

def transition_func(state, action):
    # Update location and time
    travel_stats = travel_data.stat_travel_time(
                        travel_data.to_pairs_list(
                            travel_data.get_travel_data(),
                            travel_data.get_locations()))
    curr_location = state.location
    next_location = action.location
    travel_time = travel_stats[(curr_location, next_location)]["mean"]
    time_of_day = state.time_of_day + travel_time

    # New request to add:
    new_request_list = state.request_history + [Request(location=next_location,
                                                        time_of_day=time_of_day,
                                                        day_of_week=state.day_of_week)]

    # Dynamics of people moving
    new_present_map = state.person_present_map
    for person in new_present_map:
        schedule = schedule_data.get_schedule_data()[person]

        if new_present_map[person] is None or not new_present_map[person]:
            # Check if person will arrive
            if "schedule" in schedule:
                # Room with fixed schedules
                intervals = util.parse_schedule(schedule["schedule"])

                # Each person's arrival to their office is modeled as a normal distribution centered around their scheduled time of arrival.
                p_intervals = (probability_interval(interval, state.time_of_day, time_of_day, arrival_stddev, True) for interval in intervals)
                for idx, prob in enumerate(p_intervals):
                    if random.random() < prob:
                        new_present_map[person] = int(sample_probability_interval(intervals[idx], state.time_of_day, time_of_day, arrival_stddev, True))
                        break
            elif "intervals-daily" in schedule:
                # Open area
                daily_intervals = schedule["intervals-daily"]
                p_second_arrival = daily_intervals / 24.0 / 60.0 / 60.0
                p_arrival = p_second_arrival * travel_time
                if random.random() < p_arrival:
                    new_present_map[person] = time_of_day
        else:
            # Remove person if appropriate

            # Scheduled offices
            if "schedule" in schedule:
                intervals = util.parse_schedule(schedule["schedule"])

                # Each person's departure from their office is modeled as a normal distribution centered around their scheduled time of departure.
                p_intervals = (probability_interval(interval, state.time_of_day, time_of_day, departure_stddev, False) for interval in intervals)
                for idx, prob in enumerate(p_intervals):
                    if random.random() < prob:
                        new_present_map[person] = None
                        break

            # Non-scheduled offices (mostly open spaces)
            elif "intervals-daily" in schedule:
                interval = util.Interval(
                                util.Time(state.time_of_day, new_present_map[person]),
                                util.Time(state.time_of_day, new_present_map[person] + schedule["duration"]))
                p_intervals = (probability_interval(interval, state.time_of_day, time_of_day, open_stddev, False) for interval in intervals)
                for idx, prob in enumerate(p_intervals):
                    if random.random() < prob:
                        new_present_map[person] = None
                        break

    # Resulting state
    return FullState(location=next_location,
                     time_of_day=time_of_day,
                     day_of_week=state.day_of_week,
                     person_present_map=new_present_map,
                     request_history=new_request_list)

def reward_func(state, action, newState):
    if(len(state.request_history) > len(newState.request_history)):
        latest_request = newState.request_history[-1]
        if(newState.person_present_map[latest_request.location]):
            loc_schedule_data = schedule_data.get_schedule_data()[latest_request.location]
            p_answer = loc_schedule_data["panswer"] if "panswer" in loc_schedule_data else 1
            return 1 if random.random() < p_answer else 0

    return 0

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
