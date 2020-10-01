"""plotter.py

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from lib.preproc import load_dataset

def plot_2d_pca(x: np.array, y: np.array):
    fig, ax = plt.subplot(1, 1, figsize=(10, 10))
    pca = PCA(n_components=2)
    pca.fit(x)
    X_pca = pca.transform(x)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='nipy_spectral')
    ax.xlabel('comp_1')
    ax.ylabel('comp_2')
    ax.title('PCA')
    plt.show()
    return ax

def plot_roc(x: np.array, y: np.array):
    fig, ax = plt.subplot(1, 1, figsize=(10, 10))
    return ax

def plot_training_hist(hist):
    fig, ax = plt.subplot(1, 1, figsize=(10, 10))
    return ax

DATA_LEGIT_PATH = ''
DATA_BH_PATH = ''
WSIZE = int(1e5)

if __name__ == '__main__':
    X_legit, y_legit = load_dataset(DATA_LEGIT_PATH, 0, WSIZE)
    X_bh, y_bh = load_dataset(DATA_BH_PATH, 1, WSIZE)
    X = np.concatenate(X_legit, X_bh, axis=0)
    y = np.concatenate(y_legit, y_bh, axis=0)
    plot_2d_pca(X, y)
    exit(0)