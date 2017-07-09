import numpy as np
import helpers as hp
from scipy.io import loadmat

def load_dataset(name):
    if name == "noisyX":
        # Load the image
        X = loadmat("datasets/X.mat")["X"]

        # Add noise
        X = X + np.random.randn(*X.shape) / 2 

        nRows, nCols = X.shape
        nNodes = nRows*nCols;
        nStates = 2;
         
        adj = np.zeros((nNodes,nNodes));
         
        # Add Down Edges
        ind = np.arange(nNodes)
        #  No Down edge for last row
        exclude = np.arange(1, nCols + 1) * nRows - 1
        #ind = np.setdiff1d(ind, exclude) 
        ind = np.setdiff1d(ind, exclude);

        adj[(ind,ind+1)] = 1;
        

        # Add Right Edges  
        ind = np.arange(nNodes)

        # No right edge for last column
        exclude = np.arange(nCols*nRows - nRows, nCols*nRows)
        ind = np.setdiff1d(ind, exclude) 
        adj[(ind,ind+nRows)] = 1;

        # Add Up/Left Edges
        adj = adj + adj.T
        #hp.plot_adj(adj)

        nodes, edges = hp.construct_graph(adj)


        # Standardize and ravel Features
        Xstd = (X - np.mean(X)) / np.std(X)
        Xstd = Xstd.ravel()

        # The last line transforms the noisy image intensities 
        # so that they have a mean of zero and a standard deviation of 1. 
        # We will use the following node potentials:

        n_pot = np.zeros((nNodes,nStates))
        n_pot[:,0] = np.exp(-1-2.5 * Xstd)
        n_pot[:,1] = 1

        # We want to use edge potentials that reflect that 
        # neighboring pixels are more likely to have the same label. 
        # Further, we expect that they are even more likely to have the same 
        # label if the difference in their intensities is small.
        # We incoporate this intuition by using the following Ising-like edge potentials
        n_edges = len(edges)

        e_pot = np.zeros((n_edges, nStates,nStates))
        for e in range(n_edges):
           n1, n2 = edges[e]
           pot_same = np.exp(1.8 + .3*1/(1+abs(Xstd[n1]-Xstd[n2])));
           e_pot[e] = np.array([[pot_same, 1], [1, pot_same]])
       
        return n_pot, e_pot, nodes, edges

    if name == "plane_infection":
        nSeats = 6
        nRows = 40
        nNodes = nSeats*nRows;
        nStates = 2
        adj = np.zeros((nNodes, nNodes));
        for r in range(nRows):
            for s in range(nSeats):
                if s < nSeats:
                    # % Passenger behind
                    adj[s + nSeats*(r-1),(s+1) + nSeats*(r-1)] = 1; 
                
                if r < nRows:
                    # % Passenger to the right
                    adj[s + nSeats*(r-1),s + nSeats*(r)] = 1;
  
        adj = adj+adj.T

        nodes, edges = hp.construct_graph(adj)
        # And then the potentials:
        alpha = .5;
        beta = 2;

        n_pot = np.hstack([np.ones((nNodes, 1)), alpha * np.ones((nNodes, 1))])
        n_edges = len(edges)

        e_pot = np.repeat(np.array([[[beta, 1],[1, beta]]]), n_edges, axis=0) 
        return n_pot, e_pot, nodes, edges
    if name == "bus_queue":
        n_nodes = 13
        n_states = 25

        adj = np.zeros((n_nodes, n_nodes));
        for i in range(n_nodes):
            adj[i, i+1] = 1;
        
        adj[n_nodes] = 1
        adj = adj + adj.T

        nodes, edges = hp.construct_graph(adj)
        
        busy = [10, 8, 0,3,
            5,
            4,
            0,
            5,
            0,
            0,
            0,
            0,
            0];
        n_pot = np.zeros((n_nodes, n_states));
        for n in range(n_nodes):
           for s in range(n_states):
              n_pot[n,s] = np.exp(-(1/10)*(busy[n]-(s-1))**2);
       
        n_edges = len(edges)
        e_pot = np.zeros((n_edges, n_states, n_states))
        for s1 in range(n_states):
            for s2 in range(n_states):
                e_pot[:, s1, s2] = np.exp(-(1/100)*(s1-s2)**2);
  
        return n_pot, e_pot, nodes, edges

    if name == "CS_grad_conditional":
        n_nodes = n_years = 60
        n_states = 7

        # 1. CREATE ADJACENCY MATRIX 
        adj = np.zeros((n_years, n_years), float)
        for j in range(n_years-1):
            adj[j, j+1] = 1

        adj += adj.T
        #hp.plot_adj(adj)

        # 2. CREATE GRAPH
        nodes, edges = hp.construct_graph(adj)
        n_edges = len(edges)

        # 3. Node potentials
        node_pot = np.zeros((n_nodes, n_states))
        node_pot[0] = np.array([.3 ,.6 ,.1 ,0,0, 0, 0])
        node_pot[1:] = 1

        # 4. CREATE EDGE POTENTIALS
        transitions = np.array([[.08, .9, .01, 0, 0, 0, .01],
                                     [.03, .95, .01, 0, 0, 0, .01],
                                     [.06, .06, .75, .05, .05, .02, .01],
                                     [0, 0, 0, .3, .6, .09, .01],
                                     [.0, 0, 0, .02, .95, .02, .01],
                                     [0, 0, 0, .01, .01, .97, .01],
                                     [0, 0, 0, 0, 0, 0, 1]])
    
    
        edge_pot = np.zeros((n_edges, n_states, n_states))

        for edge in range(n_edges):
            edge_pot[edge] = transitions


        observed = np.zeros(n_nodes)
        observed[9] = 6

        return node_pot, edge_pot, nodes, edges, observed

    if name == "water_turbidity":
        adj = loadmat("datasets/waterSystem.mat")["adj"]
        n_nodes = adj.shape[0]
        n_states = 4
        nodes, edges = hp.construct_graph(adj)

        n_edges = len(edges)
        source = 3
        n_pot = np.ones((n_nodes, n_states))
        n_pot[source] = [0.9, 0.09, 0.009, 0.001]

        transition = np.array([[0.9890, 0.0099, 0.0010, 0.0001],
                                [0.1309,    0.8618,    0.0066,    0.0007],
                                [0.0420,    0.0841,    0.8682,    0.0057],
                                [0.0667,    0.0333,    0.1667,    0.7333]])
        colored = np.zeros(n_nodes)
        colored[source] = 1
        done = 0
        e_pot = np.zeros((n_edges, n_states, n_states)) 

        while not done:
            done = 1
            colored_old = colored.copy()

            for e in range(n_edges):
                se_nodes = np.array(edges[e])

                if np.sum(colored_old[se_nodes]) == 1:
                    # Determine direction of edge and color nodes
                    if colored[se_nodes[0]] == 1:
                        e_pot[e] = transition
                    else:
                        e_pot[e] = transition.T

                    colored[se_nodes] = 1
                    done = 0
        
        return n_pot, e_pot, nodes, edges

    if name == "chain_CSGrads":
        n_nodes = n_years = 60
        n_states = 7

        # 1. CREATE ADJACENCY MATRIX 
        adj = np.zeros((n_years, n_years), float)
        for j in range(n_years-1):
            adj[j, j+1] = 1

        adj += adj.T
        #hp.plot_adj(adj)

        # 2. CREATE GRAPH
        nodes, edges = hp.construct_graph(adj)
        n_edges = len(edges)

        # 3. Node potentials
        node_pot = np.zeros((n_nodes, n_states))
        node_pot[0] = np.array([.3 ,.6 ,.1 ,0,0, 0, 0])
        node_pot[1:] = 1

        # 4. CREATE EDGE POTENTIALS
        transitions = np.array([[.08, .9, .01, 0, 0, 0, .01],
                                     [.03, .95, .01, 0, 0, 0, .01],
                                     [.06, .06, .75, .05, .05, .02, .01],
                                     [0, 0, 0, .3, .6, .09, .01],
                                     [.0, 0, 0, .02, .95, .02, .01],
                                     [0, 0, 0, .01, .01, .97, .01],
                                     [0, 0, 0, 0, 0, 0, 1]])
    
    
        edge_pot = np.zeros((n_edges, n_states, n_states))

        for edge in range(n_edges):
            edge_pot[edge] = transitions


        return node_pot, edge_pot, nodes, edges

    if name == "simple_studentScores":
        n_students = 4
        n_states = 2

        # 1. CREATE ADJACENCY MATRIX 
        adj = np.zeros((n_students, n_students), float)
        for j in range(n_students-1):
            adj[j, j+1] = 1
            adj[j+1, j] = 1

        # 2. CREATE GRAPH
        nodes, edges = hp.construct_graph(adj)
        n_edges = len(edges)


        # 3. Node potentials
        node_pot = np.array([[1, 3], [9, 1], [1, 3], [9, 1]], float)

        # 4. CREATE EDGE POTENTIALS
        edge_pot = np.zeros((n_edges, n_states, n_states))

        # Nodes connected by an edge have double the potential that they get the same value
        # 4 states for each edge: (0,0), (0,1), (1,0), (1,1)
        for e in range(n_edges):           
            edge_pot[e] = [[2, 1], [1, 2]]

        return node_pot, edge_pot, nodes, edges

    elif name == "ising":
        # Name Ising
        n_samples = 2500
        n_features = 2500
    
        A = np.zeros((n_samples, n_features))
    
        for i in range(n_samples):
            A[i, i] = np.random.rand()
    
            if i >= 1:
                A[i, i - 1] = np.random.randn()
            if i + 1 < n_samples:
                A[i, i + 1] = np.random.randn()
    
            if i >= 50:
                A[i, i - 50] = np.random.randn()
    
            if i + 50 < n_samples:
                A[i, i + 50] = np.random.randn()
        
        A = A.dot(A.T)
        
        w = np.random.rand(n_features)
        b = A.dot(w) + np.random.randn(n_samples)
        
    elif name == "small_ising":
        # Name Ising
        n_samples = 500
        n_features = 500
    
        A = np.zeros((n_samples, n_features))
    
        for i in range(n_samples):
            A[i, i] = np.random.rand()
    
            if i >= 1:
                A[i, i - 1] = np.random.randn()
            if i + 1 < n_samples:
                A[i, i + 1] = np.random.randn()
    
            if i >= 50:
                A[i, i - 50] = np.random.randn()
    
            if i + 50 < n_samples:
                A[i, i + 50] = np.random.randn()
        
        A = A.dot(A.T)
        
        w = np.random.rand(n_features)
        b = A.dot(w) + np.random.randn(n_samples)        
        
    elif name == "exp1":
        # l2- regularized sparse least squares
        data = loadmat("data/exp1.mat")
        A, b = data['X'], data['y']
        
    elif name == "exp2":
        # l2- regularized sparse logistic regression
        data = loadmat("data/exp2.mat")
        A, b = data['X'], data['y']
   
    elif name == "exp3":
        # Over-determined dense least squares
        data = loadmat("data/exp3.mat")
        A, b = data['X'], data['y']
        
    elif name == "exp4":
        # L1 - regularized underdetermined sparse least squares
        data = loadmat("data/exp4.mat")
        A, b = data['X'], data['y']

    elif name == "classification":
        n_samples = 2500
        n_features = 1000
        
        A, b = make_classification(n_samples, n_features)

        
    elif name == "regression":    
        n_samples = 2500
        n_features = 1000
        
        A, b = make_regression(n_samples, n_features)

    elif name == "small_reg":    
        n_samples = 250
        n_features = 250
        
        A, b = make_regression(n_samples, n_features)

    return A, b