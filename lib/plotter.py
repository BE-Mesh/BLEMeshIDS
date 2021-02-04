"""plotter.py

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from lib.preproc import load_dataset_folder, load
from lib.print_confusion_matrix import pretty_plot_confusion_matrix
from lib.trainer import stack_data, generate_model_01, BATCH_SIZE
import os
import tensorflow as tf
import pandas as pd

WSIZE = 2

IMAGES_DIR = 'images/'


def plot_histogram(y: np.array, num_classes=3, window_size: int = WSIZE):
    fig, ax = plt.subplots()
    ax.hist(y, bins=num_classes)
    plt.title(f'Classes balance (time window: {window_size} s)')
    plt.savefig(f'{IMAGES_DIR}{window_size}/hist_classes_balance.png')


def plot_2d_pca(x: np.array, y: np.array, window_size: int = WSIZE):
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)
    pca.fit(x)
    X_pca = pca.transform(x)
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='prism')
    ax.set_xlabel('comp_1')
    ax.set_ylabel('comp_2')
    ax.set_title(f'PCA (time window: {window_size} s)')
    plt.savefig(f'{IMAGES_DIR}{window_size}/pca.png')
    return ax


def plot_roc(model, data: tf.data.Dataset, num_classes=3, window_size: int = WSIZE):
    fig, ax = plt.subplots()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # todo(Fixed) inserire predict_proba
    #   inserendo adesso predict_proba vengono (0. e 1.)
    #   questa cosa altera completamente i dati della roc.
    #   Controllare i seguenti link:
    #   https://stackoverflow.com/questions/45011328/predict-proba-does-not-output-probability
    #   https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
    #
    for x_test, y_test in data.take(2):
        pred = model.predict(x_test)
        y_pred = pred[:, 1] + pred[:, 2]
        #y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    y_test = (y_test >= 1).astype(float)
    #y_pred = (y_pred >= 1).astype(int)

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='lime', linestyle=':', linewidth=4)

    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve 0 (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic (time window: {window_size} s)')
    plt.legend(loc="lower right")
    plt.savefig(f'{IMAGES_DIR}{window_size}/roc.png')
    return ax


def plot_training_hist(path: str, window_size: int):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    # Read data
    acc = np.loadtxt(os.path.join(path, 'acc.txt'))
    val_acc = np.loadtxt(os.path.join(path, 'val_acc.txt'))
    loss = np.loadtxt(os.path.join(path, 'loss.txt'))
    val_loss = np.loadtxt(os.path.join(path, 'val_loss.txt'))
    plt.suptitle(f'Training history (time window: {window_size} s)')

    axs[0].set_title('Accuracy')
    axs[0].plot(range(acc.shape[0]), acc, label='acc')
    axs[0].plot(range(acc.shape[0]), val_acc, label='val_acc')
    axs[0].legend(loc='lower right')

    axs[1].set_title('Loss')
    axs[1].plot(range(acc.shape[0]), loss, label='loss')
    axs[1].plot(range(acc.shape[0]), val_loss, label='val_loss')
    axs[1].set_yticks(np.arange(0, 10, 0.01))
    axs[1].set_yscale('log')

    axs[1].legend()

    plt.grid(True, which='both')
    plt.savefig(f'{IMAGES_DIR}{window_size}/training_hist.png')
    return axs


def plot_confusion_mat(model, data: tf.data.Dataset, num_classes=3, window_size: int = WSIZE):
    fig, ax = plt.subplots()
    cm = np.zeros(shape=(num_classes, num_classes))
    for x_val, y_val in data.take(15):
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(model.predict(x_val), axis=1)
        tcm = confusion_matrix(y_true, y_pred, labels=range(0, num_classes))
        cm += tcm
    print('row = what they are, col = what the classifier predicted')
    print(cm)
    pretty_plot_confusion_matrix(pd.DataFrame(cm), window_size=window_size, save_directory=IMAGES_DIR)

    return ax


LOGS_DIR = 'logs/'
DATA_DIR = 'data/proc'
DATA_LEGIT_PATH = f'{DATA_DIR}legit/0.csv'
DATA_BH_PATH = f'{DATA_DIR}black_hole/0.csv'


def plot(window_size: int = WSIZE):
    print('Start folder creation... ', end='')
    path = f'{IMAGES_DIR}{window_size}'
    if not os.path.exists(path=IMAGES_DIR):
        os.mkdir(path=IMAGES_DIR)
    if not os.path.exists(path=path):
        os.mkdir(path=path)
    print('done')
    print('Start loading of model... ', end='')
    X_lst, y_lst = load(DATA_DIR)
    X, y = stack_data(X_lst, y_lst, onehot=False, num_classes=3)
    plot_training_hist(LOGS_DIR, window_size=window_size)
    # onehot=False if print PCA and histogram
    plot_histogram(y, window_size=window_size)
    plot_2d_pca(X, y, window_size=window_size)
    X, y = stack_data(X_lst, y_lst, onehot=True, num_classes=3)
    X = normalize(X, axis=1, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Convert to tf.data.Dataset
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_data = train_data.batch(BATCH_SIZE).repeat().cache()
    test_data = test_data.batch(BATCH_SIZE).repeat()
    model = generate_model_01(BATCH_SIZE, num_classes=3)
    model.load_weights(f'{LOGS_DIR}weights.h5')
    print('done')
    print('Generating plots...')
    plot_training_hist(LOGS_DIR, window_size=window_size)
    plot_roc(model, test_data, window_size=window_size)
    plot_confusion_mat(model, test_data, window_size=window_size)


if __name__ == '__main__':
    plot()
    exit(0)
