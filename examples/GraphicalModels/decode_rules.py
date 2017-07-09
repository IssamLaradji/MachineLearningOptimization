"""
The decoding task is to find the most likely configuration.
"""    
import helpers as hp
import numpy as np  

def decode(n_pot, e_pot, V, E, rule="exact"):
    n_states = n_pot.shape[1]
    n_nodes = n_pot.shape[0]


    code = np.zeros(n_nodes, int)

    if rule == "viterbi":
        # Forward pass
        alpha = np.zeros((n_nodes, n_states))
        kappa = np.zeros(n_nodes)
        best_state = np.zeros((n_nodes, n_states))


        alpha[0] = n_pot[0];
        kappa[0] = np.sum(alpha[0])

        alpha[0] /= kappa[0]
        
        for i in range(1, n_nodes):
            alpha_rep = np.repeat(alpha[[i-1]], n_states, axis=0)
            tmp = alpha_rep * e_pot[i-1]

            alpha[i] = n_pot[i] * np.max(tmp, axis=1)

            best_state[i] = np.argmax(tmp, axis=1)
    
            # Normalize Message
            kappa[i] = np.sum(alpha[i])
            alpha[i] /= kappa[i]

        # Backward pass
        node_labels = np.zeros(n_nodes)
        node_labels[n_nodes - 1] = np.argmax(alpha[n_nodes-1]);

        for i in range(n_nodes - 2, -1, -1):
           node_labels[i] = best_state[i+1, node_labels[i+1]]
         
        
        return node_labels

    elif rule == "exact":
        ### Get number of evaluations
        n_evals =  n_nodes ** n_states
        print "# evaluations %d" % n_evals
        bs = -1
        bc = None
        epoch = 1
        while True:
            score = hp.compute_conf_potential(code, n_pot, e_pot, V, E)
            if epoch % 1 == 0:
                print "%d/%d: code: %s - score: %s" % (epoch, n_evals, code, score)

            if score > bs:
                bs = score
                bc = code.copy()

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
       
        return (bs, bc)

    elif rule == "ICM":
        n_nodes, n_states = n_pot.shape
        n_edges = e_pot.shape[0]

        y_pred = np.argmax(n_pot, axis=1)

        done = False
        while not done:
            done = True

            y_old = y_pred.copy()

            for n in range(n_nodes):
                # compute node potential
                pot = n_pot[n]

                # get neighbors
                edges = V[n]

                # Multiply Edge Potentials
                for e in edges:
                    s_node, e_node = E[e]

                    if n == s_node:
                        ep = e_pot[e, :, y_pred[e_node]].T
                    else:
                        ep = e_pot[e, y_pred[s_node], :]
                    
                    pot *= ep
                                
                # Assign maximum state
                y_new = np.argmax(pot)

                if y_new != y_pred[n]:
                    y_pred[n] = y_new
                    done = False 

            print "changes: %d" % np.sum(y_old != y_pred)

        return y_pred


    elif rule == "GraphCut":
        nNodes, nStates = n_pot.shape
        nEdges = e_pot.shape[0]
        
        # Supports binary graphs only
        assert nStates == 2

        # Make energies
        nodeEnergy = - np.log(n_pot)
        edgeEnergy = ee = - np.log(e_pot)
        #ee[3, 1, 1] = 1000 
        # Assert sub-modularity condition
        assert np.all(ee[:, 0, 0] + ee[:, 1, 1] <= ee[:, 0, 1] + ee[:, 1, 0], axis=0)

        # Move energy from edges to nodes
        for e in range(nEdges):
            n1, n2 = E[e]
            nodeEnergy[n1, 1] += edgeEnergy[e, 1, 0] - edgeEnergy[e, 0, 0]
            nodeEnergy[n2, 1] += edgeEnergy[e, 1, 1] - edgeEnergy[e, 0, 1]
        
        # Make Graph
        sCapacities = np.zeros(nNodes);
        tCapacities = np.zeros(nNodes);
        ndx = nodeEnergy[:,0] < nodeEnergy[:, 1]
        nd = np.logical_not(ndx)
        sCapacities[ndx] = nodeEnergy[ndx,1] - nodeEnergy[ndx,0];
        tCapacities[nd] = nodeEnergy[nd,0] - nodeEnergy[nd,1];

        eCapacities = (edgeEnergy[:, 0, 1] + edgeEnergy[:, 1, 0]
                       - edgeEnergy[:, 1,1] - edgeEnergy[:, 0,0])

        print('minCap = (%f,%f,%f)\n' % (np.min(sCapacities), 
            np.min(tCapacities),np.min(eCapacities)))
        eCapacities = np.maximum(0, eCapacities);

        # solve max flow    
        f_s = np.zeros(nNodes);
        f_e = np.zeros((nEdges, 2));
        f_t = np.zeros(nNodes);

        # To speed things up, initialize flows that don't go through edges
        for n in range(nNodes):
            cf = min(sCapacities[n], tCapacities[n]);
            if cf > 0:
                print('Initializing Direct Flow through %d\n' % n);
                
                f_s[n] = cf;
                f_t[n] = cf;
                     
        while True:
            # Compute Residual Network
            g_s = sCapacities - f_s
            g_e = np.stack([eCapacities, np.zeros(nEdges)], axis=1) - f_e            
            g_t = tCapacities - f_t;

            # Find an Augmenting Path in Residual Network
            expanded = np.zeros(nNodes);
            Q = np.where(g_s > 0)[0].tolist()

            expanded[g_s > 0] = 1;
            path = 0;
            traceBack = np.ones((nNodes,3), "int")*-1

             
            while len(Q) > 0:
                n = Q.pop(0) 
                # Check if we have an augmenting path to sink
                if g_t[n] > 0:
                    # We have found an augmenting path
                    print('Path Found');
                    path = n
                    break;

                # Check if we can push flow along one of n's edges
                edges = V[n]
                for e in edges:
                    n1, n2 = E[e] 

                    if n == n1:
                        if g_e[e,0] > 0 and not expanded[n2]:
                            # Add Neighbor to Q
                            print('Adding %d to list\n' % n2);
                            
                            expanded[n2] = 1;
                            Q += [n2]
                            traceBack[n2] = [n, e, 1];
                    else:
                        if g_e[e,1] > 0 and not expanded[n1]:
                            # Add Neighbor to Q
                            print('Adding %d to list\n',n1);
                            expanded[n1] = 1;
                            Q += [n1];
                            traceBack[n1] = [n, e, 2];

            if path:
                print('Path Found during BFS\n');
                n = path

                if traceBack[n, 0] == -1:
                    print('Direct path from source to %d to sink\n' % n)
                    raise
                else:
                    print('Indirect path from source to sink through edge\n');

                    # Compute capacity of flow in residual network
                    n = path;
                    cf = g_t[n];
                    while True:
                        if traceBack[n,0] == -1:
                            cf = min(cf, g_s[n])
                            break;
                        else:
                            # % Forward Edge
                            if traceBack[n, 2] == 1: 
                                print('Pushing flow forward\n');
                                
                                e = traceBack[n,1];
                                cf = min(cf, g_e[e,0]);
                            else:
                                print('Pushing flow backwards\n');
                                
                                e = traceBack[n,1]
                                cf = min(cf, g_e[e, 1]);
                
                            n = traceBack[n, 0];
                        
                    # Update flows
                    n = path;
                    f_t[n] += cf;
                    while True:
                        if traceBack[n, 0] == -1:
                            f_s[n] += cf;
                            break;
                        else:
                            # Forward Edge
                            if traceBack[n,2] == 1:
                                e = traceBack[n,1];
                                f_e[e,0] += cf;
                                f_e[e,1] = -f_e[e,0]
                            else:
                                e = traceBack[n, 1];
                                f_e[e,1] += cf;
                                f_e[e,0] = -f_e[e,1]
                            
                            n = traceBack[n, 0]
                  
            else:
                print('No Augmenting Path found during BFS\n');   
                break

        # Compute Min-Cut: nodes that can be reached by S in residual network
        node_labels = -expanded + 2;
        
        return node_labels

    elif rule == "belief_propagation":
        n_edges = e_pot.shape[0]
        n_states = n_pot.shape[1]
        n_nodes = n_pot.shape[0]

        n_neighbors = np.zeros(n_nodes)
        sent = np.zeros(n_edges * 2)
        waiting = np.ones(n_edges * 2)
        messages = np.zeros((n_edges * 2, n_states))

        # Compute neghbors of each node
        for i in range(n_nodes):
            n_neighbors[i] = len(V[i])

        # Get leafs as initial Queue
        Queue = np.where(n_neighbors == 1)[0].tolist()
        maximize = 1
        while len(Queue) > 0:
            node = Queue.pop(0)

            edges = V[node]

            wait = waiting[edges]
            sending = sent[edges]

            n_waiting = np.sum(wait == 1)

            # No edges are waiting
            if n_waiting == 0:
                # send final messages
                for s_edge in V[node]:
                    # sending edge
                    if sent[s_edge] == 0:
                        messages, waiting, nei = message_send(node, s_edge, n_pot, e_pot,
                                                              messages, waiting, V, E, maximize)
                        sent[s_edge] = 1
                        if n_neighbors[nei] == 1 or n_neighbors[nei] == 0:
                            Queue += [nei]
            else:
                # remaining edge
                r_edge = V[node][0]

                if sent[r_edge] == 0:
                    messages, waiting, nei = message_send(node, r_edge, n_pot, e_pot,
                                                          messages, waiting, V, E, maximize)
                    
                    sent[r_edge] = 1
                    n_neighbors[nei] -= 1

                    if n_neighbors[nei] == 1 or n_neighbors[nei] == 0:
                        Queue += [nei]

        # Once we get the messages we compute the node beliefs
        n_belief = n_pot.copy()

        for n in range(n_nodes):
            for e in V[n]:
                s_node, e_node = E[e]
                if n == s_node: 
                    n_belief[i] *= messages[e + n_edges]
                else:
                    n_belief[i] *= messages[e]

        n_belief /= np.sum(n_belief, axis=1)[:, np.newaxis]

        node_labels = np.argmax(n_belief, axis=1)

        return node_labels




def message_send(node, edge, n_pot, e_pot, messages, waiting, V, E, maximize):
    n_nodes, n_states = n_pot.shape
    n_edges = e_pot.shape[0]

    s_node, e_node = E[edge]

    if node ==  s_node:
        nei = e_node
    else:
        nei = s_node

    # Opposite node no longer waiting
    for e in V[nei]:
        if e != edge:
            waiting[e] = 0

    # Compute product of node potential with all 
    # incoming messages excpt alone `edge`
    tmp = n_pot[node]
    neighbors = V[node]

    for e in neighbors:
        if e != edge:
            s_node, e_node = E[e]

            if node == e_node:
                tmp *= messages[e]
            else:
                tmp *= messages[e + n_edges]

    s_node, e_node = E[edge]

    if node == e_node:
        pot_ij = e_pot[edge]
    else:
        pot_ij = e_pot[edge].T

    if maximize:
        new_message = hp.max_multM(pot_ij, tmp)
    else:
        new_message = pot_ij * tmp

    #normalize
    new_message /= np.sum(new_message)

    if node == e_node:
        messages[edge + n_edges] = new_message
    else:
        messages[edge] = new_message

    return messages, waiting, nei

