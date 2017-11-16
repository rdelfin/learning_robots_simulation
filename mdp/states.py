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
