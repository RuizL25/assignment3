import random

class GridworldMDP:
    def __init__(self, board, transition_model="deterministic"):
        self.board = board
        self.nrows = len(board)
        self.ncols = len(board[0])
        self.initial_state = (0, 0)
        self.transition_model = transition_model

        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.board[r][c] == 'S':
                    self.initial_state = (r, c)
                    break

        self.dr = {'up': -1, 'down': 1, 'left': 0, 'right': 0}
        self.dc = {'up': 0, 'down': 0, 'left': -1, 'right': 1}
        self.actions_list = ['up', 'down', 'left', 'right']

    def get_states(self):
        states = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.board[r][c] != '#':
                    states.append((r, c))
        return states

    def is_terminal(self, state):
        r, c = state
        cell_content = self.board[r][c]
        return cell_content not in ['', ' ', 'S', '#']

    def get_possible_actions(self, state):
        if self.is_terminal(state):
            return ['exit']
        if self.board[state[0]][state[1]] == '#':
            return []
        return self.actions_list

    def get_transitions(self, state, action):
        if self.is_terminal(state):
            if action == 'exit':
                reward = float(self.board[state[0]][state[1]])
                return [(state, 1.0, reward)]
            return []

        if self.board[state[0]][state[1]] == '#':
            return []

        transitions = []
        r, c = state

        if self.transition_model == "deterministic":
            dr, dc = self.dr[action], self.dc[action]
            next_r, next_c = r + dr, c + dc

            if not (0 <= next_r < self.nrows and 0 <= next_c < self.ncols) or self.board[next_r][next_c] == '#':
                next_state = state
            else:
                next_state = (next_r, next_c)
            transitions.append((next_state, 1.0, self.get_reward(state, action, next_state)))

        elif self.transition_model == "uniform_0.25":
            for act in self.actions_list:
                dr, dc = self.dr[act], self.dc[act]
                next_r, next_c = r + dr, c + dc

                if not (0 <= next_r < self.nrows and 0 <= next_c < self.ncols) or self.board[next_r][next_c] == '#':
                    next_state = state
                else:
                    next_state = (next_r, next_c)
                transitions.append((next_state, 0.25, self.get_reward(state, action, next_state)))
        
        elif self.transition_model == "task3_probabilities":
            intended_action_idx = self.actions_list.index(action)
            
            outcomes_probs = {
                self.actions_list[intended_action_idx]: 0.6,
                self.actions_list[(intended_action_idx + 1) % 4]: 0.2,
                self.actions_list[(intended_action_idx - 1) % 4]: 0.1,
                'stay': 0.1
            }

            for outcome_action, prob in outcomes_probs.items():
                if outcome_action == 'stay':
                    next_r, next_c = r, c
                else:
                    dr, dc = self.dr[outcome_action], self.dc[outcome_action]
                    next_r, next_c = r + dr, c + dc
                
                if not (0 <= next_r < self.nrows and 0 <= next_c < self.ncols) or self.board[next_r][next_c] == '#':
                    next_state = state
                else:
                    next_state = (next_r, next_c)
                transitions.append((next_state, prob, self.get_reward(state, action, next_state)))
        
        else:
            raise ValueError(f"Unknown transition model: {self.transition_model}")

        return transitions

    def get_reward(self, state, action, next_state):
        if self.is_terminal(next_state):
            return float(self.board[next_state[0]][next_state[1]])
        return 0.0

    def get_initial_state(self):
        return self.initial_state

class BridgeMDP(GridworldMDP):
    def __init__(self, board, transition_model="deterministic"):
        super().__init__(board, transition_model)
