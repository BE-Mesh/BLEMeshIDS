import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


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


if __name__ == '__main__':
    common_path = "A:\\Download\\Tesi\\Dataset BLE MESH\\"  # "/home/thecave3/Scaricati/btmesh-dataset/"
    src_path = common_path + "experiment_I_rpi.csv"
    sub_path = common_path + "sub.csv"
    path = "data/experiment_I.csv"

    # df = pd.read_csv(path)
    df = pd.read_csv(src_path, sep=', ', engine='python')

    clean_df(df)

    df.to_csv(path)

    # print(df.keys())
    # print(set(list(df["src"]) + (list(df["dest"]))))
    # nodes = list(set(list(df["src"]) + (list(df["dest"]))))

    # plot_dataset(df)
