"""
We say that the inference task is to find the normalizing constant Z,
 as well as the the marginal probabilities of individual nodes taking individual states.
"""

import numpy as np  
import helpers as hp

from itertools import product


def infer(n_pot, e_pot, V, E, rule="exact"):
    n_states = n_pot.shape[1]
    n_nodes = n_pot.shape[0]
    n_edges = e_pot.shape[0]

    code = np.zeros(n_nodes, int)

    if rule == "forward-backward":
        # Forward pass
        alpha = np.zeros((n_nodes, n_states))
        kappa = np.zeros(n_nodes)

        alpha[0] = n_pot[0];
        kappa[0] = np.sum(alpha[0])

        alpha[0] /= kappa[0]
        
        for i in range(1, n_nodes):
            node_prob = alpha[[i-1]]
            event_prob = e_pot[i-1]

            next_state_prob = np.dot(node_prob, event_prob)

            # Weigh next state probs with node-potentials
            alpha[i] = n_pot[i] * next_state_prob
    
            # Normalize Message
            kappa[i] = np.sum(alpha[i])
            alpha[i] /= kappa[i]
        print   
        normalizing_constant = np.product(kappa)         

        # BACKWARD PASS
        beta = np.zeros((n_nodes, n_states))
        beta[-1] = 1

        for i in range(n_nodes-2, -1, -1):
            node_prob = n_pot[[i+1]]
            event_prob = e_pot[i]

            prev_state_prob = np.dot(node_prob, event_prob.T).ravel()

            # Weigh by beliefs
            beta[i] = prev_state_prob * beta[i+1]

            # normalize
            beta[i] /= np.sum(beta[i])

        
            
        # compute node beliefs
        node_beliefs = np.zeros((n_nodes, n_states))
        for i in range(n_nodes):
            tmp = alpha[i] * beta[i]
            node_beliefs[i] = tmp / np.sum(tmp)

        print node_beliefs
        # compute edge beliefs
        edge_beliefs = np.zeros((n_edges, n_states, n_states))
        tmp = np.zeros((n_states, n_states))

        for n in range(n_nodes-1):
            for i in range(n_states):
                for j in range(n_states):
                    tmp[i, j] = alpha[n, i] * n_pot[n+1, j] * \
                                beta[n+1, j] * e_pot[n, i, j]


            edge_beliefs[n] = tmp / np.sum(tmp)
        
        # compute parition function Z
        logZ = np.sum(np.log(kappa))

    if rule == "mean_field" or rule =="mf":
        nNodes, nStates = n_pot.shape
        nEdges = e_pot.shape[0]

        nodeBel = n_pot.copy()
        for n in range(nNodes):
            nodeBel[n] /= np.sum(nodeBel[n])


        maxIter = 100
        # Compute node beliefs
        for i in range(maxIter):
            nodeBel_old = nodeBel.copy()
            for n in range(nNodes):
                b = np.zeros(nStates)

                # Get neighbors
                edges = V[n]
                for e in edges:
                    n1, n2 = E[e]
                    if n1 == n:
                        # forward edge
                        ep = e_pot[e]
                        neigh = n2
                    else:
                        # backward edge
                        ep = e_pot[e].T
                        neigh = n1

                    for s in range(nStates):
                        b[s] += np.dot(nodeBel[neigh], np.log(ep[:, s]))

                nb = n_pot[n] * np.exp(b);
                nodeBel[n] = nb / np.sum(nb); 

            diff = np.linalg.norm(nodeBel - nodeBel_old)
            print "diff %.3f" %  diff
            if diff < 1e-4:
                break

        # Compute edge beliefs    
        edgeBel = np.zeros(e_pot.shape);
        for e in range(nEdges):
            n1, n2 = E[e]
            for s1, s2 in product(range(nStates), range(nStates)):
                edgeBel[e, s1, s2] = nodeBel[n1, s1] * nodeBel[n2,s2]
                
        
        return nodeBel, edgeBel, 0.
            
        
        # Normalizing factor
        #
   
    if rule == "loopybeliefpropagation" or rule =="lbp" or rule=="trbp":
        nNodes, nStates = n_pot.shape
        nEdges = e_pot.shape[0]

        maximize = 0
        if rule == "lbp":
            msg_new = utils.UGM_loopyBP(n_pot, e_pot, V, E, maximize)
        elif rule == "trbp":
            msg_new = utils.UGM_TRBP(n_pot, e_pot, V, E, maximize, True)

        # Compute nodeBel
        prod_of_msgs = np.zeros((nNodes, nStates))

        nodeBel = np.zeros(n_pot.shape)

        for n in range(nNodes):
            edges = V[n]
            prod_of_msgs[n] = n_pot[n]
            
            for e in edges:
                n1, n2 = E[e]
                if n == n1:
                    prod_of_msgs[n] *= msg_new[e]
                else:
                    prod_of_msgs[n] *= msg_new[e+nEdges]
                
            
            nodeBel[n] = prod_of_msgs[n] / np.sum(prod_of_msgs[n])
        
        return nodeBel, None, None

    if rule == "exact":
        ### Get number of evaluations
        n_evals =  n_nodes ** n_states
        print "# evaluations %d" % n_evals
        bs = -1
        bc = None
        epoch = 1


        # Initialize vertex and edge marginal probabilities
        v_mar = np.zeros(n_pot.shape)
        e_mar = np.zeros(e_pot.shape)
        Z = 0.

        while True:
            pot = hp.compute_conf_potential(code, n_pot, e_pot, V, E)
            if epoch % 1 == 0:
                print "%d/%d: code: %s - score: %s" % (epoch, n_evals, code, pot)

            for v in range(n_nodes):
                v_mar[v, code[v]] += pot 

            for e in range(n_edges):
                (v1, v2) = E[e]                
                e_mar[e, code[v1], code[v2]] += pot

            Z += pot
            # Go to next y
            for v in range(n_nodes):
                code[v] +=  1

                if code[v] < n_states:
                    break
                else:
                    code[v] = 0

            if v == (n_nodes - 1) and code[-1] == 0:
                break
            
            epoch += 1
       
        return v_mar / Z, Z




