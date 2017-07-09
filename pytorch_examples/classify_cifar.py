from os import sys, path

import numpy as np
from sklearn.utils import shuffle

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import dataset_utils as du
import pytorch_kit.models as tm

if __name__ == "__main__":
    # LOAD, SHUFFLE, AND SPLIT DATA
    X, y = du.load_dataset("cifar")
    X, y = shuffle(X, y)

    # GET Y LABELS
    y = np.argmax(y, axis=1)

    n = X.shape[0]
    Xtest, ytest = X[n/2:], y[n/2:]
    X, y = X[:n/2], y[:n/2]
    
    # INITIALIZE AND TRAIN MODEL
    model = tm.cifar_net(problem_type="classification",
                         n_channels=X.shape[1], 
                         n_classes=np.unique(y).size)

    model.fit(X, y, epochs=10, batch_size=50)

    # REPORT SCORE
    print("Training score: %.3f" % model.score(X, y))
    print("Test score: %.3f" % model.score(Xtest, ytest))
