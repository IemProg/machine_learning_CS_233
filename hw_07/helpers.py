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

# helper for Cross Validation'''
def train_val_split(num_examples,ratio):
    ind = np.arange(num_examples)
    np.random.shuffle(ind)
    train_ind = ind[:-int(num_examples*ratio)]
    val_ind = ind[-int(num_examples*ratio):]
    return train_ind, val_ind

from sklearn.preprocessing import StandardScaler
# it's good to standarize our data. Helps in faster convergence.
scaler = StandardScaler()

# Function for using kth split as validation set to get accuracy
# and k-1 splits to train our model
def do_cross_validation(clf,k,k_fold_ind,X,Y):
    
    # use one split as val
    val_ind = k_fold_ind[k]
    # use k-1 split to train
    train_splits = [i for i in range(k_fold_ind.shape[0]) if i is not k]
    train_ind = k_fold_ind[train_splits,:].reshape(-1)
    
    #Get train and val 
    X_train = X[train_ind,:]
    Y_train = Y[train_ind]
    X_val = X[val_ind,:]
    Y_val = Y[val_ind]
    
    scaler.fit(X_train)
    X_train = scaler.fit_transform(X_train)
    #fit the train transformation on val
    X_val = scaler.fit_transform(X_val)
    
    #fit on train set
    clf.fit(X_train,Y_train)
    #get accuracy for val
    acc = clf.score(X_val,Y_val)
    return acc

# Function to split data indices
# num_examples: total samples in the dataset
# k_fold: number fold of CV
# returns: array of shuffled indices with shape (k_fold, num_examples//k_fold)
def fold_indices(num_examples,k_fold):
    ind = np.arange(num_examples)
    split_size = num_examples//k_fold
    
    #important to shuffle your data
    np.random.seed(42)
    np.random.shuffle(ind)
    
    k_fold_indices = []
    #CODE HERE#
    # Generate k_fold set of indices
    k_fold_indices = [ind[k*split_size:(k+1)*split_size] for k in range(k_fold)]
         
    return np.array(k_fold_indices)

'''
Input:
    X: NxD matrix representing our data
    d: Number of principal components to be used to reduce dimensionality
    
Output:
    mean_data: 1xD representing the mean of input data
    W: Dxd principal components
    eg: d values representing variance corresponding to principal components
    X_hat: Nxd data projected in principal components' direction
    exvar: explained variance by principal components
'''
def PCA(X, d):
    
    # Compute the mean of data
    mean = np.mean(X, 0)
    # Center the data with the mean
    X_tilde = X - mean
    # Create covariance matrix
    C = X_tilde.T@X_tilde/X_tilde.shape[0]
    # Compute eigenvector and eigenvalues. Hint use: np.linalg.eigh
    eigvals, eigvecs = np.linalg.eigh(C)
    # Choose top d eigenvalues and corresponding eigenvectors. Sort eigenvalues( with corresponding eigenvector )
    # in decreasing order first.
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    W = eigvecs[:, 0:d]
    eg = eigvals[0:d]

    # project the data using W
    X_hat = X_tilde@W
    
    #explained variance
    exvar = 100*eg.sum()/eigvals.sum()

    return mean, W, eg, X_hat, exvar



def kernel_rbf(xi, xj, sigma):
    """ Computes the RBF kernel function for the input vectors `xi`, `xj`.
    
    Args:
        xi (np.array): Input vector, shape (D, ).
        xj (np.array): Input vector, shape (D, ).
        
    Returns:
        float: Result of the kernel function.
    """

    return np.exp(-(1. / (2. * sigma ** 2)) * ((xi - xj) ** 2).sum())
    ######################
    
    
def kernel_matrix(X, kernel_func, *kernel_func_args):
    """ Computes the kernel matrix for data `X` using kernel 
    function `kernel_func`.
    
    Args:
        X (np.array): Data matrix with data samples of dimension D in the 
            rows, shape (N, D).
        kernel_func (callable): Kernel function.
        kernel_func_args: Arguments of the `kernel_func` to be called.
            
    Returns:
        np.array: Kernel matrix of the shape (N, N).
    """

    N, D = X.shape
    
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel_func(X[i], X[j], *kernel_func_args)
    assert np.allclose(K, K.T)
    return K
    ######################  



