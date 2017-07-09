import numpy as np 
import pylab as plt

# COMPUTE POTENTIAL h2
def compute_conf_potential(code, n_pot, e_pot, V, E):
    pot = 1.

    n_nodes = n_pot.shape[0]
    n_edges = e_pot.shape[0]

    # Score over vertices
    for v in range(n_nodes):
        pot *= n_pot[v, code[v]]

    # Score over edges
    for e in range(n_edges):
        (v1, v2) = E[e]
        pot *= e_pot[e, code[v1], code[v2]]

    return pot  

# UGM HELPERS
def UGM_loopyBP(n_pot, e_pot, V, E, maximize):
    # Initialize
    nNodes, nStates = n_pot.shape
    nEdges = e_pot.shape[0]

    # nodeBel = np.zeros((nNodes, nStates));
    # nodeBel_old = nodeBel.copy()
    # prod_of_msgs = np.zeros((nNodes, nStates))

    msg_old = np.zeros((nEdges*2, nStates))
    msg_new = np.zeros((nEdges*2, nStates))

    for e in range(nEdges):
        n1, n2 = E[e]
        # Forward Message from n1 => n2
        msg_new[e] = 1. / nStates  

        # Backward Message from n2 => n1
        msg_new[e+nEdges] = 1. / nStates 

    maxIter = 2000
    for i in range(maxIter):
        for n in range(nNodes):
            # Neighbors
            edges = E[n]
            for e in edges:
                n1, n2 = E[e]

                if n == n1:
                    ep = e_pot[e]
                else:
                    ep = e_pot[e].T

                # Compute product of incoming messages
                tmp = n_pot[n]
                for eg in edges:
                    ng1, ng2 = E[eg]

                    if eg != e:
                        if n == ng1:
                            tmp *= msg_new[eg]
                        else:
                            tmp *= msg_new[eg + nEdges]

                # Compute new message
                if maximize:
                    newm = maxMult(ep, tmp) 
                else:
                    newm = np.dot(ep, tmp)


                if n == n1:
                    msg_new[e]= newm / np.sum(newm);
                else:
                    msg_new[e + nEdges]= newm / np.sum(newm);

        diff = np.linalg.norm(msg_new - msg_old) 

        print "%d: diff %.3f" % (i, diff)
        if diff < 1e-4:
            break

        msg_old = msg_new.copy()

    return msg_new


def UGM_TRBP(n_pot, e_pot, V, E, maximize, weighted=True):
    # Initialize
    nNodes, nStates = n_pot.shape
    nEdges = e_pot.shape[0]

    if not weighted:
        return UGM_loopyBP(n_pot, e_pot, V, E, maximize)
    else:
        # Generate Random Spanning Trees until all edges are covered
        count = 0.
        edgeAppears = np.zeros(nEdges);
        while True:
            count += 1.
            np.random.seed(int(count))
            edgeAppears += minSpan(nNodes, E);
            print np.sum(edgeAppears==0)
            if all(edgeAppears > 0):
                break;

        mu = edgeAppears/count;

    msg_old = np.zeros((nEdges*2, nStates))
    msg_new = np.zeros((nEdges*2, nStates))

    for e in range(nEdges):
        n1, n2 = E[e]
        # Forward Message from n1 => n2
        msg_new[e] = 1. / nStates  

        # Backward Message from n2 => n1
        msg_new[e+nEdges] = 1. / nStates 

    maxIter = 2000
    for i in range(maxIter):
        for n in range(nNodes):
            # Neighbors
            edges = E[n]
            for e in edges:
                n1, n2 = E[e]

                if n == n1:
                    ep = e_pot[e]
                else:
                    ep = e_pot[e].T

                # Adjust edge potentials by edge appearance probability
                ep = ep**(1./mu[e])

                # Compute product of incoming messages
                tmp = n_pot[n]
                for eg in edges:
                    ng1, ng2 = E[eg]

                    if eg != e:
                        if n == ng1:
                            tmp *= msg_new[eg] ** mu[eg]
                        else:
                            tmp *= msg_new[eg + nEdges]**(1-mu[eg])

                # Compute new message
                if maximize:
                    newm = maxMult(ep, tmp) 
                else:
                    newm = np.dot(ep, tmp)


                if n == n1:
                    msg_new[e]= newm / np.sum(newm);
                else:
                    msg_new[e + nEdges]= newm / np.sum(newm);

        diff = np.linalg.norm(msg_new - msg_old) 

        print "%d: diff %.3f" % (i, diff)
        if diff < 1e-4:
            break

        msg_old = msg_new.copy()

    return msg_new


# MISC 
def minSpan(nNodes, E):
    # % [E] = minSpan(nNodes,edgeEnds)
    # %
    # % Compute minimum spanning tree using Prim's algorithm
    # %   (if graph is disconnected, repeats the procedure to generate a minimum
    # %   spanning forest)
    # %
    # % edgeEnds(e,[n1 n2 w]):
    # %   gives the two nodes and weight for each e
    nEdges = len(E)

    # Initialize with no nodes or edges included
    nodes = np.zeros(nNodes);
    edges = np.zeros(nEdges);

    # Sort edges by weight
    # [sorted, sortedInd] = sort(edgeEnds(:,3));

    while True:
        # Find those nodes not yet added
        ind = np.where(nodes == 0)[0]
        if ind.size == 0:
            break;
        
        # Randomly add an initial node (from among those not added yet)
        nR = np.random.randint(low=0, high=ind.size)
        nodes[ind[nR]] = 1
        
        done = 0;
        while not done:
            done = 1;
            
            # Find minimal weight edge s.t. V(n1) = 0 and V(n2) = 1 or vice versa
            for e in range(nEdges):
                n1, n2 = E[e]
                if nodes[n1] == 0 and nodes[n2] == 1:
                    nodes[n1] = 1;
                    edges[e] = 1;
                    done = 0;
                    break;
                elif nodes[n2] == 0 and nodes[n1] == 1:
                    nodes[n2] = 1;
                    edges[e] = 1;
                    done = 0;
                    break;

    return edges

def standardizeCols(X, tied):
    if tied:
        X.shape
        import pdb; pdb.set_trace()  # breakpoint 923ce592 //

    else:
        pass

def max_multM(A, B):
    # Just like the matrix multipliation of A and B, 
    # except we max instead of sum
    assert B.ndim == 1

    n_elements = B.size

    result = A.copy()
    for i in range(n_elements):
        result[:, i] *= B[i]

    return np.max(result, axis=1) 


# GRAPH PLOT HELPERS
def construct_graph(adj_matrix):    
    n_nodes = adj_matrix.shape[0]
    height, width = adj_matrix.shape

    assert height == width
    assert height == n_nodes

    edges = {}
    nodes = {i:[] for i in range(n_nodes)}
    edge_ids = []
    edge_count = 0

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj_matrix[i, j] != 0:
                # Add edge
                edges[edge_count] = (i, j)

                # Add edges corresponding to the two vertices
                nodes[i] += [edge_count]
                nodes[j] += [edge_count]

                # Update edge meta info
                edge_ids += [edge_count]
                edge_count += 1

    return nodes, edges

def plot_adj(adj):
    G = nx.from_numpy_matrix(adj)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))
    G = nx.drawing.nx_agraph.to_agraph(G)

    G.node_attr.update(color="red", style="filled")
    G.edge_attr.update(color="blue", width="2.0")
    # nx.draw(G)
    G.draw('tmp/out.jpg', format='jpg', prog='neato')
    #pl.draw()
    # img=mpimg.imread('tmp/out.png')
    command  = "gnome-open tmp/out.jpg"

    call(command, shell=True)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(imgplot)
    #G.draw('out.png', format='png', prog='neato')

def plot_samples(samples):
    # Sample 100 samples
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,10))
    axes = axes.ravel()
    
    for j in range(samples.shape[0]):
        axes[j].hist(samples[j], bins=2)
        axes[j].set_title("student %d" % j)

    plt.tight_layout()
    plt.show()