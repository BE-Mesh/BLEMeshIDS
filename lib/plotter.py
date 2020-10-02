"""plotter.py

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

from lib.preproc import load_dataset_folder
from lib.print_confusion_matrix import pretty_plot_confusion_matrix
from lib.trainer import stack_data, generate_model_01, BATCH_SIZE
import os
import tensorflow as tf
import pandas as pd


def plot_histogram(y: np.array, num_classes=3):
    fig, ax = plt.subplots()
    ax.hist(y, bins=num_classes)
    plt.savefig('../images/hist_classes_balance.png')


def plot_2d_pca(x: np.array, y: np.array):
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)
    pca.fit(x)
    X_pca = pca.transform(x)
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='prism')
    ax.set_xlabel('comp_1')
    ax.set_ylabel('comp_2')
    ax.set_title('PCA')
    plt.savefig('../images/pca.png')
    return ax


def plot_roc(model, data: tf.data.Dataset, num_classes=3):
    fig, ax = plt.subplots()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for x_test, y_test in data.take(2):
        y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    y_test = (y_test >= 1).astype(int)
    y_pred = (y_pred >= 1).astype(int)

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('../images/roc.png')

    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(num_classes):
    #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= num_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # # Plot all ROC curves
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(num_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()
    return ax


def plot_training_hist(path: str):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    # Read data
    acc = np.loadtxt(os.path.join(path, 'acc.txt'))
    val_acc = np.loadtxt(os.path.join(path, 'val_acc.txt'))
    loss = np.loadtxt(os.path.join(path, 'loss.txt'))
    val_loss = np.loadtxt(os.path.join(path, 'val_loss.txt'))
    plt.suptitle('Training history')

    axs[0].set_title('Accuracy')
    axs[0].plot(range(acc.shape[0]), acc, label='acc')
    axs[0].plot(range(acc.shape[0]), val_acc, label='val_acc')
    axs[0].legend()

    axs[1].set_title('Loss')
    axs[1].plot(range(acc.shape[0]), loss, label='loss')
    axs[1].plot(range(acc.shape[0]), val_loss, label='val_loss')
    axs[1].set_yticks(np.arange(0, 10, 0.01))
    axs[1].set_yscale('log')

    axs[1].legend()

    plt.grid(True, which='both')
    plt.savefig('../images/training_hist.png')
    return axs


def plot_confusion_mat(model, data: tf.data.Dataset, num_classes=3):
    fig, ax = plt.subplots()
    cm = np.zeros(shape=(num_classes, num_classes))
    for x_val, y_val in data.take(15):
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(model.predict(x_val), axis=1)
        tcm = confusion_matrix(y_true, y_pred, labels=range(0, num_classes))
        cm += tcm
    print('row = what they are, col = what the classifier predicted')
    print(cm)
    pretty_plot_confusion_matrix(pd.DataFrame(cm))

    return ax


DATA_LEGIT_PATH = '../data/legit/0.csv'
DATA_BH_PATH = '../data/black_hole/0.csv'

if __name__ == '__main__':
    X_lst, y_lst = load_dataset_folder('../data/')
    X, y = stack_data(X_lst, y_lst, onehot=False, num_classes=3)
    plot_training_hist('../logs')
    # onehot=False if print PCA and histogram
    plot_histogram(y)
    plot_2d_pca(X, y)
    X, y = stack_data(X_lst, y_lst, onehot=True, num_classes=3)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.33, random_state=42)
    # Convert to tf.data.Dataset
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_data = train_data.batch(BATCH_SIZE).repeat().cache()
    test_data = test_data.batch(BATCH_SIZE)
    model = generate_model_01(BATCH_SIZE, num_classes=3)
    model.load_weights('../logs/weights.h5')

    plot_training_hist('../logs/')
    plot_roc(model, test_data)
    plot_confusion_mat(model, test_data)

    exit(0)
