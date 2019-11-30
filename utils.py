import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from scipy.spatial import distance
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

sns.set()


def plot_dtw_dist(x, y, figsize=(12, 12), annotation=True, col_map="PuOr"):

    '''
    Take two arrays columns array to plot the dynaic time wrapping map
    '''

    assert x.shape[1] == 1, \
        "first array needs to be a column array of shape (n,1)"

    assert y.shape[1] == 1, \
        "second array needs to be a column array of shape (m,1)"

    dist, path = fastdtw(x, y, dist=euclidean)

    n_timestamps_1 = len(x)
    n_timestamps_2 = len(y)
    matrix_path = np.zeros((n_timestamps_1, n_timestamps_2))
    for i in tuple(path)[::-1]:
        matrix_path[i] = 1

    matrix_path = np.transpose(matrix_path)[::-1]
    matrix_dist = np.transpose(distance.cdist(x, y, 'euclidean'))[::-1]

    fig, axScatter = plt.subplots(figsize=figsize)

    sns.heatmap(matrix_dist, annot=annotation, ax=axScatter,
                cbar=False, cmap=col_map)
    sns.heatmap(matrix_dist, annot=annotation, ax=axScatter,
                cbar=False, mask=matrix_path < 1,
                annot_kws={"weight": "bold"},
                cmap=sns.dark_palette((210, 90, 0), n_colors=2, input="husl"))

    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("left", 1, pad=0.1, sharey=axScatter)

    # make some labels invisible
    axScatter.xaxis.set_tick_params(labelbottom=False)
    axScatter.yaxis.set_tick_params(labelleft=False)
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHistx.yaxis.set_tick_params(labelleft=False)
    axHisty.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)

    axHistx.plot(x)
    axHisty.plot(y, range(len(y)))

    plt.show()
