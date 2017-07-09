def apply_algorithm(rule, Pi, V, R, P, n_states, n_actions, gamma):

    if rule == "value_iteration":
        # Reference: http://webdocs.cs.ualberta.ca/~sutton/book/4/node5.html
        delta = 100.
        while delta > 1e-6:
            delta = 0.
            for s in range(n_states):
                VBest = VOld = V[s]

                for a in range(n_actions):
                    Vsa = 0
                    for sNext in range(n_states):
                        Vsa +=  P[s, a, sNext] * (R[s, a, sNext] + gamma * V[sNext])

                    if Vsa > VBest:
                        VBest = Vsa

                    delta = max(delta, abs(VOld - VBest))

                V[s] = VBest

        for s in range(n_states):
            VBest = V[s]

            for a in range(n_actions):
                Vsa = 0
                for sNext in range(n_states):
                    Vsa +=  P[s, a, sNext] * (R[s, a, sNext] + gamma * V[sNext])

                if Vsa > VBest:    
                    Pi[s] = a
                    VBest = Vsa

        return Pi, V

    if rule == "policy_iteration":
        # Reference: http://webdocs.cs.ualberta.ca/~sutton/book/4/node4.html

        # Evaluate policy: repeat until convergence
        delta = 100.
        while delta > 1e-6:        
            delta = 0.
            for s in range(n_states):
                v = 0
                
                for sNext in range(n_states):
                    v += P[s, Pi[s], sNext] * (R[s, Pi[s], sNext] + gamma * V[sNext])

                delta = max(delta, abs(v - V[s]))
                V[s] = v

        # Improve policy
        for s in range(n_states): 
            VBest = V[s]

            for a in range(n_actions):
                Vsa = 0
                for sNext in range(n_states):
                    Vsa +=  P[s, a, sNext] * (R[s, a, sNext] + gamma * V[sNext])

                if Vsa > VBest:    
                    Pi[s] = a
                    VBest = Vsa

        return Pi, V