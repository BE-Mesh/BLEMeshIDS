import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA


def nodes_connected(g, u, v):
    return u in g.neighbors(v)


def plot_keras_model(model, filename: str = 'model.png'):
    tf.keras.utils.plot_model(
        model, to_file=filename, show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )


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


def plot_2d_pca(df: pd.DataFrame, experiment_name: str, time_window: int):
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


def plot_model_accuracy(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_model_loss(history):
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

