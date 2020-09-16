import random

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow import keras

from DataLoader.data_loader import data_loader
from DataParser.data_parser import preprocessing_phase
import tensorflow as tf


def clean_df(dataframe):
    del dataframe["experiment"]
    del dataframe["setting"]
    del dataframe["run"]
    del dataframe["source node"]
    del dataframe["node"]


def nodes_connected(g, u, v):
    return u in g.neighbors(v)


def save_cleaned_copy(dataframe):
    clean_df(dataframe)
    dataframe.to_csv("experiment_I.csv", encoding='utf-8')


def plot_dataset(dataframe):
    # dataframe = dataframe.drop(dataframe[dataframe.dest == "ffff"].index)  # broadcast
    dataframe = dataframe.drop(dataframe[dataframe.src == "0000"].index)  # ??
    dataframe = dataframe.drop(dataframe[dataframe.dest == "0000"].index)  # ??

    G: nx.DiGraph = nx.from_pandas_edgelist(dataframe, 'src', 'dest', create_using=nx.DiGraph)
    # print(len(G.nodes()))
    # print(len(G.edges()))

    # Plot it
    nx.draw_networkx(G, arrows=True)

    # print(nodes_connected(G, "010e", "0102"))

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_labels(G, pos)
    # nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)  # edgelist=edges,
    plt.show()


def plot_2d_pca(df: pd.DataFrame, experiment_name: str):
    result_np = df.to_numpy()
    pca = PCA(2)  # target dimension
    projected = pca.fit_transform(result_np)
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title(f'PCA of Dataset {experiment_name}')
    plt.suptitle(f"time windows: {time_window} ms")
    plt.colorbar()
    plt.show()


def plot_3d_pca(df: pd.DataFrame):
    pca = PCA(n_components=3)
    pca.fit(df)

    # Store results of PCA in a data frame
    result = pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(3)], index=df.index)

    # Plot initialisation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap="Set2_r", s=60)

    # make simple, bare axis lines through space:
    xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0, 0), (min(result['PCA2']), max(result['PCA2'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA")
    plt.show()
    # rotate
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)


if __name__ == '__main__':
    labels_values = {"legit": 0, "bubu": 1}
    experiments = {"experiment_I_rpi.csv": 'legit', "experiment_II_lpn.csv": 'legit'}

    images: np.ndarray
    labels = []
    for i, experiment in enumerate(experiments.keys()):
        path = "data/" + experiment
        time_window = 1000000

        print(f'Started preprocessing for "{experiment}"')
        res_df = preprocessing_phase(source_path=path, t_window=time_window)
        print("Finished preprocessing")

        # print(res_df.head())
        processed_path = "data/preprocessed_" + experiment
        res_df.to_csv(processed_path, index=False)

        x, y = data_loader(processed_path, labels_values[experiments[experiment]])
        images = x if i == 0 else np.concatenate((images, x))
        labels += y

    labels = np.array(labels)
    c = list(zip(images, labels))
    random.shuffle(c)
    x, y = zip(*c)
    x = np.array(x)
    y = np.array(y)

    # for scikit learn
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # for tensorflow
    training_set = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_set = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    training_set = training_set.batch(32).cache().repeat()
    test_set = test_set.batch(32)

    cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=True)

    model.fit(training_set, epochs=600, steps_per_epoch=15, validation_data=test_set, callbacks=[cb_es])

    # plot_2d_pca(res_df)
    # plot_3d_pca(res_df)

    # kmeans = KMeans(n_clusters=2, random_state=0).fit(res_df)
    # print(kmeans.get_params())
