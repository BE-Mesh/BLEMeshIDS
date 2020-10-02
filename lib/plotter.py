"""plotter.py

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

from lib.preproc import load_dataset_folder
from lib.trainer import stack_data, generate_model_01, BATCH_SIZE
import os
import tensorflow as tf


def plot_histogram(y: np.array, num_classes=2):
    fig, ax = plt.subplots()
    ax.hist(y, bins=num_classes)
    plt.show()


def plot_2d_pca(x: np.array, y: np.array):
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)
    pca.fit(x)
    X_pca = pca.transform(x)
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='bwr')
    ax.set_xlabel('comp_1')
    ax.set_ylabel('comp_2')
    ax.set_title('PCA')
    plt.show()
    return ax


def plot_roc(model, data: tf.data.Dataset, num_classes=2):
    fig, ax = plt.subplots()
    fpr = dict()
    tpr = dict()
    for x_test, y_test in data.take(2):
        y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
    for i in range(num_classes):
        ax.plot(fpr[i], tpr[i], label=f'Class {i}')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')
    plt.show()
    return ax


def plot_training_hist(path: str):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    # Read data
    acc = np.loadtxt(os.path.join(path, 'acc.txt'))
    val_acc = np.loadtxt(os.path.join(path, 'val_acc.txt'))
    loss = np.loadtxt(os.path.join(path, 'loss.txt'))
    val_loss = np.loadtxt(os.path.join(path, 'val_loss.txt'))
    axs[0].plot(range(acc.shape[0]), acc, label='acc')
    axs[0].plot(range(acc.shape[0]), val_acc, label='val_acc')
    axs[1].plot(range(acc.shape[0]), loss, label='loss')
    axs[1].plot(range(acc.shape[0]), val_loss, label='val_loss')
    plt.show()
    return axs


def plot_confusion_mat(model, data: tf.data.Dataset, num_classes=2):
    fig, ax = plt.subplots()
    cm = np.zeros(shape=(num_classes, num_classes))
    for x_val, y_val in data.take(15):
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(model.predict(x_val), axis=1)
        tcm = confusion_matrix(y_true, y_pred, labels=range(0, num_classes))
        cm += tcm
    print(cm)

    return ax


def plot_roc_curve(y_score, n_classes: int, target_curve: int = 2):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[target_curve], tpr[target_curve], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[target_curve])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    target_exp = ['legit', 'black hole', 'grey hole']
    plt.title(f'Receiver operating characteristic for {target_exp[target_curve]} detection')
    plt.legend(loc="lower right")
    plt.show()


DATA_LEGIT_PATH = '../data/legit/0.csv'
DATA_BH_PATH = '../data/black_hole/0.csv'

if __name__ == '__main__':
    X_lst, y_lst = load_dataset_folder('../data/')
    X, y = stack_data(X_lst, y_lst, onehot=True)
    # plot_training_hist('../logs')
    # plot_histogram(y)
    # plot_2d_pca(X, y)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.33, random_state=42)
    # Convert to tf.data.Dataset
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_data = train_data.batch(BATCH_SIZE).repeat().cache()
    test_data = test_data.batch(BATCH_SIZE)
    model = generate_model_01(BATCH_SIZE)
    model.load_weights('../logs/weights.h5')

    plot_roc(model, test_data)
    # plot_confusion_mat(model, test_data)

    exit(0)
