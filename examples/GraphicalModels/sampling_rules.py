"""
The sampling task is to simulate a set of configurations, 
according to the probability of them occurring in the UGM.
"""

import numpy as np
import helpers as hp

def sample(n_samples, v_pot, e_pot, V, E):
    n_states = v_pot.shape[1]
    n_vertices = v_pot.shape[0]

    # Compute normalizing factor
    Z = compute_Z(v_pot, e_pot, V, E)
    samples = np.zeros((n_samples, n_vertices))

    for i in range(n_samples):
        random = np.random.rand()
        code = np.zeros(n_vertices, int)
        cum_pot = 0. 
        while True:
            pot = hp.compute_conf_potential(code, v_pot, e_pot, V, E)
            cum_pot += pot

            if cum_pot / Z > random:
                samples[i] = code

                break

            # Go to next y
            for v in range(n_vertices):
                code[v] +=  1

                if code[v] < n_states:
                    break
                else:
                    code[v] = 0

            if v == (n_vertices - 1) and code[-1] == 0:
                break

    return samples


def compute_Z(v_pot, e_pot, V, E):
    """ Compute the normalizing factor"""
    n_vertices = v_pot.shape[0]
    n_states = v_pot.shape[1]
    
    Z = 0.
    code = np.zeros(n_vertices, int)

    while True:
        pot = hp.compute_conf_potential(code, v_pot, e_pot, V, E)

        Z += pot
        # Go to next y
        for v in range(n_vertices):
            code[v] +=  1

            if code[v] < n_states:
                break
            else:
                code[v] = 0

        if v == (n_vertices - 1) and code[-1] == 0:
            break

   
    return Z