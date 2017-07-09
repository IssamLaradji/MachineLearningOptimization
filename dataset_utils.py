import numpy as np 
import os
from sklearn.datasets import fetch_mldata
from sklearn.datasets import make_regression, make_classification
#from keras.datasets import cifar10
from sklearn.utils import shuffle as skShuffle
import torchvision 
import glob
import pandas as pd 
from skimage import img_as_float
from skimage.io import imread
import image_utils as iu


def load_dataset(name, as_image=False, data_home="../../Datasets/"):

    if name == "mnist_small":
        mnist = fetch_mldata('MNIST original', data_home=data_home)

        A = mnist["data"].astype(float) / 255.
        b = mnist["target"].astype(int) 
        n_classes = np.unique(b).size
        b = to_categorical(b, n_classes)
        if as_image:
            A = np.reshape(A, (A.shape[0], 1, 28,28))
        A, b = skShuffle(A, b)
        
        return A[:2000], np.argmax(b, axis=1)[:2000]
        return train_valid_test_split(A, b)

    if name == "mnist":
        mnist = fetch_mldata('MNIST original', data_home=data_home)

        A = mnist["data"].astype(float) / 255.
        b = mnist["target"].astype(int) 
        n_classes = np.unique(b).size
        b = to_categorical(b, n_classes)
        if as_image:
            A = np.reshape(A, (A.shape[0], 1, 28,28))
        A, b = skShuffle(A, b)

        return A, np.argmax(b, axis=1)
        return train_valid_test_split(A, b)

    if name == "cifar":

        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True)
        x, y = trainset.train_data, trainset.train_labels
        x_test, y_test = testset.test_data, testset.test_labels

        x = iu.last2first(x)
        x_test = iu.last2first(x_test)

        #y = np.array(y)
        #y_test = np.array(y_test)
        y = to_categorical(y, 10)
        y_test = to_categorical(y_test, 10)

        x = x.astype(float) / 255.
        x_test = x_test.astype(float) / 255.

        data = train_valid_split(x, y)
        data["Xtest"] = x_test
        data["ytest"] = y_test

        return data["Xtrain"], data["ytrain"]

    if name == "boat_images":
        # LOAD SMALL IMAGES
        imgList = glob.glob("boat_images/*.png")
        df = pd.read_csv("boat_images/coords.csv")

        X = np.zeros((len(imgList), 80, 80, 3))
        Y = np.zeros((len(imgList), 80, 80))
        for i, img in enumerate(imgList):
            X[i] = img_as_float(imread(img))

            flag = False
            yx_coors = []
            for _, row in df.iterrows():
                if img[img.rindex("/")+1:] == row.image[row.image.rindex("/")+1:]:
                    yx_coors += [(row.y, row.x)]
                    flag = True
            if flag == False:
                Y[i] = np.zeros((80, 80))
            else:
                #Y[i] =  np.ones((80, 80))*-1

                for y, x in yx_coors:
                    Y[i, y, x] = 1

        X = iu.last2first(X)
        Y = Y[:, np.newaxis]

        if as_image:
            return X, Y
        else:
            y = Y.sum(axis=1).sum(axis=1).sum(axis=1)
      

            return X, y.astype(int)


    elif name == "2class":
        n_samples = 2500
        n_features = 1000
        
        A = generate_A(n_samples, n_features)

        x = np.random.randn(n_features)

        b = np.sign(np.dot(A, x))
        
        b = b.ravel()

    elif name == "5class":
        n_samples = 2500
        n_features = 1000
        n_classes = 5
        
        A = generate_A(n_samples, n_features)
        b = np.random.randint(0, n_classes, n_samples)
        b = to_categorical(b, n_classes)   

    elif name == "5classSparse":
        data = load_dataset(data_home + "exp2")
        A = data["A"]
        b = data["b"]
        n_samples = A.shape[0]
        n_classes = 5

        b = np.random.randint(0, n_classes, n_samples)
        b = to_categorical(b, n_classes)   


    elif name == "50class":
        n_samples = 2500
        n_features = 1000
        n_classes = 50
        
        A = generate_A(n_samples, n_features)
        b = np.random.randint(0, n_classes, n_samples)
        b = to_categorical(b, n_classes)   

    elif name == "sparse60":
        np.random.seed(1)
        n_samples = 2500
        n_features = 1000
        
        A = generate_A(n_samples, n_features)

        x = np.random.randn(n_features)
        b = np.sign(np.dot(A, x))
        
        M = np.zeros(A.shape)
        n_cols = M.shape[1]

        for i in range(A.shape[0]):
            n_ones = np.random.randint(3,6)
            indices = np.random.choice(n_cols, size=n_ones, replace=False)
            M[i, indices] = 1

        A = M * A
        b = b.ravel()
        btmp = b.copy()
        btmp[btmp!=1] = -1

        # Q = np.outer(b, b) * np.dot(A, A.T)
        left = np.dot(np.diag(btmp), A) 
        right = np.dot(A.T, np.diag(btmp))
        Q = np.dot(left, right)

        assert (Q>0).sum(axis=1).max() == 60
 

    elif name == "sparse30":
        n_samples = 2500
        n_features = 1000
        
        A = generate_A(n_samples, n_features)

        x = np.random.randn(n_features)
        b = np.sign(np.dot(A, x))
        
        M = np.zeros(A.shape)
        n_cols = M.shape[1]

        for i in range(A.shape[0]):
            n_ones = np.random.randint(15,30)
            indices = np.random.choice(n_cols, size=n_ones, replace=False)
            M[i, indices] = 1

        A = M * A
        b = b.ravel()


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def generate_A(n_samples, n_features):
    A = np.random.randn(n_samples, n_features)
    A += 1

    return A
    
def train_valid_test_split(X, y, shuffle=True):
    np.random.seed(1)
    
    if shuffle:
        X, y = skShuffle(X, y)

    n = X.shape[0]

    # SPLIT BETWEEN TEST AND TRAIN
    cut = int(n*0.9)
    Xtest, ytest = X[cut:], y[cut:]
    X, y = X[:cut], y[:cut]
    
    # SPLIT BETWEEN VALID AND TRAIN
    n = X.shape[0]
    cut = int(n*0.9)
    Xtrain, ytrain = X[:cut], y[:cut]
    Xvalid, yvalid = X[cut:], y[cut:]

    return {"Xtrain":Xtrain, "ytrain":ytrain, 
            "Xvalid":Xvalid, "yvalid":yvalid,
            "Xtest":Xtest, "ytest":ytest}

def train_valid_split(X, y, shuffle=True):
    np.random.seed(1)
    
    if shuffle:
        X, y = skShuffle(X, y)

    n = X.shape[0]

    # SPLIT BETWEEN VALID AND TRAIN
    cut = int(n*0.9)
    Xtrain, ytrain = X[:cut], y[:cut]
    Xvalid, yvalid = X[cut:], y[cut:]

    return {"Xtrain":Xtrain, "ytrain":ytrain, 
            "Xvalid":Xvalid, "yvalid":yvalid}
