import numpy as np

def convert(action):
    if action == 0:
        return "^"
    if action == 1:
        return ">"
    if action == 2:
        return "v"
    if action == 3:
        return "<"
        

class GridWorld:
    def __init__(self, n_rows=5, n_cols=5):
        self.n_states = n_rows * n_cols
        self.n_actions = 4

        self.n_rows = n_rows
        self.n_cols = n_cols

        # Construct Rewards with 4 terminals 
        # 2 good terminals and 2 bad ones
        rewards = np.zeros((self.n_states, self.n_actions, self.n_states))
        rewards[:, :, n_cols-1] = 1
        rewards[:, :, -1] = 1

        rewards[:, :, 2*n_cols - 2] = -1
        rewards[:, :, n_rows/3*n_cols + n_cols/2] = -1

        # Construct Transition matrix
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                succ_states = self.get_successive_states(s, a)
                for sN in succ_states:
                    P[s, a, sN] = 1./3

        self.probability_transition_matrix = P
        self.rewards = rewards

        # Display grid world in terms of reward
        print np.reshape(rewards[0, 0], (n_rows, n_cols))

    def step(self, action):

        succ_states = self.get_successive_states(self.state, action)

        # Generate a random number from the set {0,1,2} 
        rand = np.random.randint(3)
        # Get one of the three successive states with equal probability
        state_next = succ_states[rand]

        reward = self.rewards[self.state, action, state_next]

        done = False
        if reward == 0:
            # Gets penalty for living
            reward = - 0.05
        else:
            # Reached a terminal state
            done = True

        self.state = state_next

        return state_next, reward, done, None

    def reset(self):
        self.state = 0

        return self.state


    def get_successive_states(self, state, action):
        row = state / self.n_cols 
        col = state % self.n_cols

        succ_states = [0, 0, 0]

        if action == 0:
            # Go up - results in going up, left or right with equal prob.
            succ_states = [(row - 1, col),
                           (row, col - 1),
                           (row, col + 1)] 
        if action == 1:
            # Go right - results in going right, up or down with equal prob.
            succ_states = [(row, col + 1),
                           (row - 1, col),
                           (row + 1, col)] 

        if action == 2:
            # Go down - results in going down, left or right with equal prob.
            succ_states = [(row + 1, col),
                           (row, col - 1),
                           (row, col + 1)] 

        if action == 3:
            # Go left - results in going left, up or down with equal prob.
            succ_states = [(row, col - 1),
                           (row - 1, col),
                           (row + 1, col)] 

        
        def get_state(row_col):
            row, col = row_col

            # Make sure row and col are within boundaries
            row = min(row, self.n_rows - 1)
            row = max(row, 0)
            col = min(col, self.n_cols - 1)
            col = max(col, 0)

            return row * self.n_cols + col


        succ_states = map(get_state, succ_states)

        return succ_states

