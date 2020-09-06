import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def clean_df(df):
    del df["experiment"]
    del df["setting"]
    del df["run"]
    del df["source node"]
    del df["node"]


def nodes_connected(G, u, v):
    return u in G.neighbors(v)


def save_cleaned_copy(df):
    clean_df(df)
    df.to_csv("experiment_I.csv", encoding='utf-8')


if __name__ == '__main__':
    common_path = "/home/thecave3/Scaricati/btmesh-dataset/"  # "A:\\Download\\Tesi\\Dataset BLE MESH\\"
    src_path = common_path + "experiment_I_rpi.csv"
    sub_path = common_path + "sub.csv"
    path = "experiment_I.csv"

    # df = pd.read_csv(path)
    df = pd.read_csv(src_path, sep=', ', engine='python')

    # print(df.keys())
    # print(set(list(df["src"]) + (list(df["dest"]))))
    #
    # nodes = list(set(list(df["src"]) + (list(df["dest"]))))

    # print(nodes)

    # save_cleaned_copy(df)

    df = df.drop(df[df.dest == "fffd"].index)  # broadcast
    df = df.drop(df[df.src == "0000"].index)  # ??
    df = df.drop(df[df.dest == "0000"].index)  # ??

    G: nx.DiGraph = nx.from_pandas_edgelist(df, 'src', 'dest', create_using=nx.DiGraph)
    # print(len(G.nodes()))
    # print(len(G.edges()))

    # Plot it
    # nx.draw(G, with_labels=True))

    nx.draw_networkx(G, arrows=True)

    # print(nodes_connected(G, "010e", "0102"))

    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'))
    # nx.draw_networkx_labels(G, pos)
    # nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', arrows=True)

    plt.show()
