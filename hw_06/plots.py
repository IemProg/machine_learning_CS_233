import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotC(fracts, nDims, trials):
    plt.figure()
    domain = np.round(np.linspace(2,200,25)).astype(int)/nDims
    plt.plot(domain,fracts)
    plt.xlabel("p/N")
    plt.ylabel("C(p,N)")
    plt.title("Fraction of convergences per {} trials as a function of p".format(trials))

def plot3Dscatter(X):
    prefix = "original" if X.shape[1]<3 else "transformed"
    if X.shape[1]<3:
        Z = np.zeros((X.shape[0],1))
        X = np.hstack([X, Z])
        
    fig = plt.figure()
    ax  = Axes3D(fig)
    point_size = 1000
    colors_vec = ["red","blue","blue","red"]
    ax.scatter(list(X[:,0]), list(X[:,1]), list(X[:,2]), s=point_size, c=colors_vec)
    ax.set_zlim3d(0,2)

    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("z")
    plt.title("XOR problem {} data points".format(prefix))


'''Plotting helper for SVM exercise'''
def plot(X,Y,clf,show=True,dataOnly=False):
    
    plt.figure()
    # plot data points
    Y[Y==-1] = 0
    X1 = X[Y==0]
    X2 = X[Y==1]
    Y1 = Y[Y==0]
    Y2 = Y[Y==1]
    class1 = plt.scatter(X1[:, 0], X1[:, 1], zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)
    class2 = plt.scatter(X2[:, 0], X2[:, 1], zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)
    if not dataOnly:
        # get the range of data
        x_min = X[:, 0].min() 
        x_max = X[:, 0].max() 
        y_min = X[:, 1].min() 
        y_max = X[:, 1].max()

        # sample the data space
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

        # apply the model for each point
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)

        # plot the partitioned space
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        
        # plot hyperplanes
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '--'], levels=[-1, 0, 1], alpha=0.5)
        
        # plot support vectors
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                edgecolors='g', s=100, linewidth=1)
#         plt.xlim(left=x_min-0.3)
#         plt.ylim(top=y_max+0.3)
    if dataOnly:
        plt.title('Data Set')
    else:
        if clf.kernel == 'rbf':
            plt.title('Decision Boundary and Margins, C={}, gamma={}'.format(clf.C,clf.gamma)) 
        elif clf.kernel == 'poly':
            plt.title('Decision Boundary and Margins, C={}, degree={}'.format(clf.C,clf.degree)) 
        else:
            plt.title('Decision Boundary and Margins, C={}'.format(clf.C)) 
        
    plt.legend((class1,class2),('Claas A','Class B'),scatterpoints=1,
           loc='best',
           ncol=3,
           fontsize=8)
     
    if show:
        plt.show()
    Y[Y==0] = -1
        
'''Plotting Heatmap for CV results'''
def plot_cv_result(grid_val,grid_search_c,grid_search_gamma):
    plt.figure(figsize=(8,10))
    plt.imshow(grid_val)
    plt.colorbar()
    plt.xticks(np.arange(len(grid_search_gamma)), grid_search_gamma, rotation=20)
    plt.yticks(np.arange(len(grid_search_c)), grid_search_c, rotation=20)
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.title('Val Accuracy for different Gamma and C')
    plt.show()
    
def plot_simple_data():
    #Data set
    x_neg = np.array([[2,4],[1,4],[2,3]])
    y_neg = np.array([-1,-1,-1])
    x_pos = np.array([[6,-1],[7,-1],[5,-3]])
    y_pos = np.array([1,1,1])
    x1 = np.linspace(-10,10)
    x = np.vstack((np.linspace(-10,10),np.linspace(-10,10)))

    #Parameters guessed by inspection
    w = np.array([1,-1]).reshape(-1,1)
    b = -3

    #Plot
    fig = plt.figure(figsize = (6,6))
    plt.scatter(x_neg[:,0], x_neg[:,1], marker = 'x', color = 'r', label = 'Negative -1')
    plt.scatter(x_pos[:,0], x_pos[:,1], marker = 'o', color = 'b',label = 'Positive +1')
    plt.plot(x1, x1  - 3, color = 'darkblue')
    plt.plot(x1, x1  - 7, linestyle = '--', alpha = .3, color = 'b')
    plt.plot(x1, x1  + 1, linestyle = '--', alpha = .3, color = 'r')
    plt.xlim(0,10)
    plt.ylim(-5,5)
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(-5, 5, step=1))

    #Lines
    plt.axvline(0, color = 'black', alpha = .5)
    plt.axhline(0,color = 'black', alpha = .5)
    plt.plot([2,6],[3,-1], linestyle = '-', color = 'darkblue', alpha = .5 )
    plt.plot([4,6],[1,1],[6,6],[1,-1], linestyle = ':', color = 'darkblue', alpha = .5 )
    plt.plot([0,1.5],[0,-1.5],[6,6],[1,-1], linestyle = ':', color = 'darkblue', alpha = .5 )

    #Annotations
    plt.annotate(s = '$A \ (6,-1)$', xy = (5,-1), xytext = (6,-1.5))
    plt.annotate(s = '$B \ (2,3)$', xy = (2,3), xytext = (2,3.5))#, arrowprops = {'width':.2, 'headwidth':8})
    plt.annotate(s = '$2$', xy = (5,1.2), xytext = (5,1.2) )
    plt.annotate(s = '$2$', xy = (6.2,.5), xytext = (6.2,.5))
    plt.annotate(s = '$2\sqrt{2}$', xy = (4.5,-.5), xytext = (4.5,-.5))
    plt.annotate(s = '$2\sqrt{2}$', xy = (2.5,1.5), xytext = (2.5,1.5))
    plt.annotate(s = '$w^Tx + b = 0$', xy = (8,4.5), xytext = (8,4.5))
    plt.annotate(s = '$(\\frac{1}{4},-\\frac{1}{4}) \\binom{x_1}{x_2}- \\frac{3}{4} = 0$', xy = (7.5,4), xytext = (7.5,4))
    plt.annotate(s = '$\\frac{3}{\sqrt{2}}$', xy = (.5,-1), xytext = (.5,-1))

    #Labels and show
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc = 'lower right')
    plt.show()
