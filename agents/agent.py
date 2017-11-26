class Agent:
    def next_action(self, state):
        """
        Method should return the agent's next action given
        the state
        """
        raise NotImplementedError
    
    def action_update(self, reward, new_state):
        raise NotImplementedError
