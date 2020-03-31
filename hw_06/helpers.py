import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# code to fit perceptron from exercise sheet 3

def predict(x, w):
    """predict value of observation"""
    return ((x.dot(w) >= 0.0) - 0.5) * 2.0

def Perceptron(X, y, w, lr=0.1, n_epochs=50):
    """kernelized perceptron"""
    # in case the target labels should be {0,1} then remap to {-1,1}
    y[y==0]     = -1
    num_samples = X.shape[0]

    for ep in range(n_epochs):
        # draw indices randomly
        idxs = np.random.permutation(num_samples)
        
        num_errs = 0 # count errors within current epoch
        for idx in idxs:

            # check whether the current observation was classified correctly
            correct = predict(X[idx], w) == y[idx]
            
            # Update weights
            if not correct:
                num_errs += 1
                w += lr * X[idx] * y[idx]
        
        # stopping criterion
        if num_errs == 0: break
    return w,num_errs

# helper for SVM exercise

''' Gives a simple toy dataset.'''
def get_simple_dataset():
    # create a toy dataset
    np.random.seed(1)
    X, Y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 10))
    return X,Y

'''Gives two circluar dataset'''
def get_circle_dataset():
    np.random.seed(0)
    X,Y = datasets.make_circles(n_samples=100, factor=.5,
                                      noise=.05)
    return X,Y

'''Iris Dataset Loader Function'''
def get_iris_dataset():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_2d = X[:, ::2]
    X_2d = X_2d[y > 0]
    y_2d = y[y > 0]
    y_2d -= 1

    return X_2d, y_2d

'''Create the splits for Cross Validation'''
def train_val_split(num_examples,ratio):
    ind = np.arange(num_examples)
    np.random.shuffle(ind)
    train_ind = ind[:-int(num_examples*ratio)]
    val_ind = ind[-int(num_examples*ratio):]
    return train_ind, val_ind


