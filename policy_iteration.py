import random
from collections import defaultdict

class PolicyIteration:
    def __init__(self, mdp, discount=0.9, iterations=100, evaluation_iterations=50):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.evaluation_iterations = evaluation_iterations
        self.values = defaultdict(float)
        self.policy = {}

    def _compute_qvalue(self, state, action, current_values) -> float:
        q_value = 0.0
        transitions = self.mdp.get_transitions(state, action)
        for next_state, prob, reward in transitions:
            q_value += prob * (reward + self.discount * current_values.get(next_state, 0.0))
        return q_value

    def policy_evaluation(self, policy, initial_values=None) -> dict:
        values = defaultdict(float, initial_values if initial_values is not None else {})

        for _ in range(self.evaluation_iterations):
            new_values = values.copy()
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    new_values[state] = 0.0
                    continue

                action = policy.get(state)
                if action is None:
                    new_values[state] = 0.0 
                    continue

                new_values[state] = self._compute_qvalue(state, action, values)
            
            values = new_values
            
        return values

    def policy_improvement(self, current_values) -> dict:
        new_policy = {}
        for state in self.mdp.get_states():
            if self.mdp.is_terminal(state):
                new_policy[state] = 'exit'
                continue

            possible_actions = self.mdp.get_possible_actions(state)
            if not possible_actions:
                new_policy[state] = None
                continue

            q_values_for_state = {action: self._compute_qvalue(state, action, current_values) for action in possible_actions}
            best_action = max(possible_actions, key=lambda action: q_values_for_state[action])
            new_policy[state] = best_action
        return new_policy

    def run_policy_iteration(self):
        self.policy = {state: random.choice(self.mdp.get_possible_actions(state)) 
                       for state in self.mdp.get_states() if not self.mdp.is_terminal(state)}
        for state in self.mdp.get_states():
            if self.mdp.is_terminal(state):
                self.policy[state] = 'exit'
            elif state not in self.policy and self.mdp.get_possible_actions(state):
                 self.policy[state] = random.choice(self.mdp.get_possible_actions(state))
            elif state not in self.policy:
                 self.policy[state] = None


        for iteration in range(self.iterations):
            self.values = self.policy_evaluation(self.policy, initial_values=self.values)

            new_policy = self.policy_improvement(self.values)
            
            policy_converged = True
            for state in self.mdp.get_states():
                if self.policy.get(state) != new_policy.get(state):
                    policy_converged = False
                    break
            
            self.policy = new_policy

            if policy_converged:
                print(f"Policy converged after {iteration + 1} iterations.")
                break
        
    def get_value(self, state) -> float:
        return self.values.get(state, 0.0)

    def get_policy(self, state):
        return self.policy.get(state, None)

    def get_qvalue(self, state, action):
        return self._compute_qvalue(state, action, self.values)
