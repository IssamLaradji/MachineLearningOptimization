import pylab as plt
import numpy as np
import dataset_utils as du
import decode_rules as dr 
import inference_rules as ir 
import sampling_rules as sr 
import argparse
import sys
import helpers as hp


if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--demo', required=True) 

    io_args = parser.parse_args()
    demo = io_args.demo

    if demo == "small":
        n_pot, e_pot, V, E = du.load_dataset("simple_studentScores")

        # Decode: Compute optimal decoding (most likely configuration of the states)
        print dr.decode(n_pot, e_pot, V, E, rule="exact")

        # Infer: Compute Vertex and Edge marginals and Normalizing Constant
        print ir.infer(n_pot, e_pot, V, E, rule="exact")

        # Sample:
        dep_samples = sr.sample(100, n_pot, e_pot, V, E).T

        # Display New sampling results
        #ind_samples = sample_independent_potentials(node_pot)
        hp.plot_samples(dep_samples)

    elif demo == "chain":
        n_pot, e_pot, V, E = du.load_dataset("chain_CSGrads")

        # Decode: Compute optimal decoding (most likely configuration of the states)
        print dr.decode(n_pot, e_pot, V, E, rule="viterbi")

        # Infer: Compute Vertex and Edge marginals and Normalizing Constant
        print ir.infer(n_pot, e_pot, V, E, rule="forward-backward")

        # Sample:
        dep_samples = sr.sample(100, n_pot, e_pot, V, E).T

        # Display New sampling results
        #ind_samples = sample_independent_potentials(node_pot)


        hp.plot_samples(dep_samples)

    elif demo == "tree":
        n_pot, e_pot, V, E = du.load_dataset("water_turbidity")

        # Decode: Compute optimal decoding (most likely configuration of the states)
        print dr.decode(n_pot, e_pot, V, E, rule="belief_propagation")

        # Decode: Compute Vertec and Edge marginals and normalizing constant
        print dr.decode(n_pot, e_pot, V, E, rule="belief_propagation")


    elif demo == "conditional":
        n_pot, e_pot, V, E, observed = du.load_dataset("CS_grad_conditional")

    elif demo == "cutset":
        n_pot, e_pot, V, E = du.load_dataset("bus_queue")

    elif demo == "junction":
        n_pot, e_pot, V, E = du.load_dataset("plane_infection")

    elif demo == "graph_cut":
        n_pot, e_pot, V, E = du.load_dataset("noisyX")
        labels_original = np.argmax(n_pot, axis=1)
        labels_original = np.reshape(labels_original, (32, 32))

        # Decode: Compute approximate decoding with icm
        labels_icm = dr.decode(n_pot, e_pot, V, E, rule="ICM")
        labels_icm = np.reshape(labels_icm, (32, 32))

        # # Decode: Compute optimal decoding with graphcut
        labels_gc = dr.decode(n_pot, e_pot, V, E, rule="GraphCut")
        labels_gc = np.reshape(labels_gc, (32, 32))

        plt.subplot(131)
        plt.title("Original image")
        plt.imshow(labels_original)

        plt.subplot(132)
        plt.title("ICM image decoding")
        plt.imshow(labels_icm)
 
        plt.subplot(133)
        plt.title("Graph Cut image decoding")
        plt.imshow(labels_gc)
        plt.show()       

    elif demo == "meanField":
        n_pot, e_pot, V, E = du.load_dataset("noisyX")

        labels_original = np.argmax(n_pot, axis=1)
        labels_original = np.reshape(labels_original, (32, 32))

        
        nBel, eBel, _ = ir.infer(n_pot, e_pot, V, E, rule="mf")
        labels = np.argmax(nBel, axis=1)
        labels = np.reshape(labels, (32, 32))


        plt.subplot(121)
        plt.title("Original image")
        plt.imshow(labels_original)
        
        plt.subplot(122)
        plt.title("meanField image decoding")
        plt.imshow(labels)
        plt.show() 


    elif demo == "lbp":
        n_pot, e_pot, V, E = du.load_dataset("noisyX")

        labels_original = np.argmax(n_pot, axis=1)
        labels_original = np.reshape(labels_original, (32, 32))

        
        nBel, eBel, _ = ir.infer(n_pot, e_pot, V, E, rule="lbp")
        labels = np.argmax(nBel, axis=1)
        labels = np.reshape(labels, (32, 32))


        plt.subplot(121)
        plt.title("Original image")
        plt.imshow(labels_original)
        
        plt.subplot(122)
        plt.title("meanField image decoding")
        plt.imshow(labels)
        plt.show() 


    elif demo == "trbp":
        n_pot, e_pot, V, E = du.load_dataset("noisyX")

        labels_original = np.argmax(n_pot, axis=1)
        labels_original = np.reshape(labels_original, (32, 32))

        
        nBel, eBel, _ = ir.infer(n_pot, e_pot, V, E, rule="trbp")
        labels = np.argmax(nBel, axis=1)
        labels = np.reshape(labels, (32, 32))


        plt.subplot(121)
        plt.title("Original image")
        plt.imshow(labels_original)
        
        plt.subplot(122)
        plt.title("trbp image decoding")
        plt.imshow(labels)
        plt.show() 