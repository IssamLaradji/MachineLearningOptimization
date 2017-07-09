def update(rule, state, reward, next_state):

    if rule == "temporal_difference":
        V[state] = alpha * (r + gamma * V[next_state] - V[state])

        return V[state]
