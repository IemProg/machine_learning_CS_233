# 3rd party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import datasets
import itertools


def load_ds_iris(sep_l=True, sep_w=True, pet_l=True, pet_w=True,
                 setosa=True, versicolor=True, virginica=True, addbias=True):
    """ Loads the iris dataset [1]. The function arguments select which
    features and classes will be included in the dataset.

    [1] https://en.wikipedia.org/wiki/Iris_flower_data_set

    Args:
        sep_l (bool): Include "sepal length" feature.
        sep_w (bool): Include "sepal width" feature.
        pet_l (bool): Include "petal length" feature.
        pet_w (bool): Include "petal width" feature.
        setosa (bool): Include "setosa" class, 50 samples.
        versicolor (bool): Include "versicolor" class, 50 samples.
        virginica (bool): Include "virginica" class, 50 samples.

    Returns:
        data (np.array of float64): Data, shape (N, D), N depends on `setosa`,
            `versicolor`, `virginica` (each gives 50 samples), D depends on
            `sep_l`, `sep_w`, `pet_l`, `pet_w` (each gives 1 feature).
        labels (np.array of int64): Labels, shape (N, ).
    """

    # Load ds.
    d, l = datasets.load_iris(return_X_y=True)
    data = np.empty((0, 4))
    labels = np.empty((0, ), dtype=np.int64)

    # Get classes.
    for idx, c in enumerate([setosa, versicolor, virginica]):
        if c:
            data = np.concatenate([data, d[l == idx]], axis=0)
            labels = np.concatenate([labels, l[l == idx]], axis=0)

    # Get features.
    feats_incl = []
    for idx, f in enumerate([sep_l, sep_w, pet_l, pet_w]):
        if f:
            feats_incl.append(idx)
    data = data[:, feats_incl]
    #data = np.concatenate([np.ones([data.shape[0],1]),data], axis=1)

    if addbias:
        data = np.concatenate((np.ones([data.shape[0],1]),data[:, feats_incl]), axis=1)

    return data, labels

def scatter2d_multiclass(data, labels, fig=None, fig_size=None, color_map=None,
                         marker_map=None, legend=True, legend_map=None,
                         grid=False, show=False, aspect_equal=False):
    """ Plots the 2D scatter plot for multiple classes.

    Args:
        data (np.array of float): Data, shape (N, 2) or (N, 3), N is # of samples of
            (x, y) coordinates.
        labels (np.array of int): Class labels, shape (N, )
        fig (plt.Figure): The Figure to plot to. If None, new Figure will be
            created.
        fig_size (tuple): Figure size.
        color_map (dict): Mapping of classes inds to string color codes.
            If None, each class is assigned different color automatically.
        maerker_map (dict): Mapping of classes inds to to string markers.
        legend (bool): Whetehr to print a legend.
        legend_map (dict): Mapping of classes inds to str class labels.
            If None, the int inds are uased as labels.
        grid (bool): Whether to show a grid.
        show (bool): Whether to show the plot.
        aspect_equal (bool): Whether to equalize the aspect ratio for the axes.

    Returns:
        plt.Figure
    """
    # Check dims.
    labels = labels.flatten()
    if data.shape[1] == 3:
        data = data[:, 1:]
    assert(data.ndim == 2 and data.shape[1] == 2)
    assert(data.shape[0] == labels.shape[0])

    # Get classes.
    classes = np.unique(labels)

    # Prepare class colors.
    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    if color_map is None:
        color_map = {}
        for cl in classes:
            color_map[cl] = next(colors)
    # assert(np.all(sorted(list(color_map.keys())) == classes))
    assert(np.all([cl in color_map.keys() for cl in classes]))

    # Prepare class markers.
    markers = itertools.cycle(['o', 'x', '+', '*', 'D', 'p', 's'])
    if marker_map is None:
        marker_map = {}
        for cl in classes:
            marker_map[cl] = next(markers)
    # assert (np.all(sorted(list(marker_map.keys())) == classes))
    assert (np.all([cl in marker_map.keys() for cl in classes]))

    # Prepare legend labels.
    if legend_map is None:
        legend_map = {}
        for cl in classes:
            legend_map[cl] = cl
    assert(np.all(sorted(list(legend_map.keys())) == classes))

    # Plots
    if fig is None:
        fig, _ = plt.subplots(1, 1, figsize=fig_size)
    ax = fig.gca()
    for cl in classes:
        ax.plot(data[:, 0][labels == cl], data[:, 1][labels == cl],
                linestyle='', marker=marker_map[cl], color=color_map[cl],
                label=legend_map[cl])

    if aspect_equal:
        ax.set_aspect('equal', adjustable='datalim')

    if legend:
        ax.legend()
    if grid:
        ax.grid()
    if show:
        fig.show()

    return fig

def generate_dataset_synth():
    np.random.seed(0)
    data_a = np.random.normal([1,1], [0.35,0.35], [50,2])
    data_b = np.random.normal([0,-1], [0.1,0.25], [50,2])
    data_c = np.random.normal([-1,0.2], [0.4,0.1], [50,2])
    data_multi = np.concatenate([data_a, data_b, data_c], axis=0)
    #data_multi = np.concatenate([np.ones([150,1]),np.concatenate([data_a, data_b, data_c], axis=0)], axis=1)
    labels_multi = np.zeros([150,])
    labels_multi[50:100] = 1
    labels_multi[100:] = 2
    np.save("data_synth", data_multi)
    np.save("labels_synth", labels_multi)

def load_dataset_synth(addbias=True):
    data_multi = np.load("data_synth.npy")
    labels_multi = np.load("labels_synth.npy")
    if addbias:
        data_multi = np.concatenate([np.ones([data_multi.shape[0],1]),data_multi], axis=1)

    onehot_labels = label_to_onehot(labels_multi)
    return data_multi, onehot_labels

def label_to_onehot(label):
    one_hot_labels = np.zeros([label.shape[0], int(np.max(label)+1)])
    one_hot_labels[np.arange(label.shape[0]), label.astype(np.int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    return np.argmax(onehot, axis=1)

def visualize_predictions(data, labels_gt, labels_pred, fig=None):
    """ Visualizes the dataset, where the GT classes are denoted by "x", "o", "+" markers which are colored according to predicted labels.

    Args:
        data (np.array): Dataset, shape (N, D).
        labels_gt (np.array): GT labels, shape (N, ).
        labels_pred (np.array): Predicted labels, shape (N, )
        fig (plt.Figure): Figure to plot to. If None, new one is created.

    Returns:
        plf.Figure: Figure.
    """
    if data.shape[1] == 3:
        data = data[:, 1:]

    if fig is None:
        fig, _ = plt.subplots(1, 1)
    ax = fig.gca()

    # Prepare legend labels.
    classes = np.unique(labels_gt)
    marker_map = {0: 'x', 1: 'o', 2: '+'}
    color_map={0: 'r', 1: 'g', 2: 'b'}
    for gt_class in classes:
        for pred_class in classes:
            points = np.logical_and(labels_gt == gt_class, labels_pred == pred_class)
            ax.plot(data[:, 0][points], data[:, 1][points],
                linestyle='', marker=marker_map[gt_class], color=color_map[pred_class],
                label="gt: " + str(gt_class) + ", pred:" + str(pred_class))
    ax.legend()

    return fig