class ValueIteration:
    
    def __init__(self, mdp, discount=0.9, iterations=100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {}

    def run_value_iteration(self):
        for _ in range(self.iterations):
            new_values = {}
            
            for state in self.mdp.get_states():
                
                if self.mdp.is_terminal(state):
                    new_values[state] = 0.0
                    continue

                actions = self.mdp.get_possible_actions(state)
                
                if not actions:
                    new_values[state] = 0.0
                else:
                    q_values = [self.compute_qvalue_from_values(state, action) for action in actions]
                    new_values[state] = max(q_values)
            
            self.values = new_values

    def get_value(self, state) -> float:
        return self.values.get(state, 0.0)

    def compute_qvalue_from_values(self, state, action) -> float:
        q_value = 0.0

        transitions = self.mdp.get_transitions(state, action)
        
        for next_state, prob, reward in transitions:
            q_value += prob * (reward + self.discount * self.get_value(next_state))
            
        return q_value

    def compute_action_from_values(self, state):

        actions = self.mdp.get_possible_actions(state)
        
        if not actions or self.mdp.is_terminal(state):
            return None
        
        if state not in self.values:
            self.values[state] = 0.0

        return max(actions, key=lambda action: self.compute_qvalue_from_values(state, action))

    def get_action(self, state):
        return self.compute_action_from_values(state)

    def get_qvalue(self, state, action):
        return self.compute_qvalue_from_values(state, action)

    def get_policy(self, state):
        return self.compute_action_from_values(state)