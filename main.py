import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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


def preprocessing_phase(source_path: str, t_window: int, clean_bcast: bool = False, ) -> pd.DataFrame:
    df = pd.read_csv(source_path)

    if clean_bcast:
        df = df.drop(df[df.dest == "ffff"].index)  # broadcast
        df = df.drop(df[df.src == "0000"].index)  # ??
        df = df.drop(df[df.dest == "0000"].index)  # ??

    # df.sample(n=500).to_csv(path)

    min_time = df["sent time"].min()
    df["time_window"] = np.floor((df["sent time"] - min_time) / t_window)
    dfs = []
    for i in range(int(df["time_window"].max())):
        window = df.loc[df["time_window"] == i]
        if not window.empty:
            temp_dict = {"size": window.size, "rssi_mean": np.mean(window["rssi"]), "rssi_std": np.std(window["rssi"]),
                         "ttl_mean": np.mean(window["ttl"]), "ttl_std": np.std(window["ttl"])}
            unique, counts = np.unique(window["src"], return_counts=True)
            pkts_src = list(dict(zip(unique, counts)).values())  # packtes per source
            temp_dict["src_mean"] = np.mean(pkts_src)
            temp_dict["src_std"] = np.std(pkts_src)
            unique, counts = np.unique(window["dest"], return_counts=True)
            pkts_dst = list(dict(zip(unique, counts)).values())  # packtes per dest
            temp_dict["dest_mean"] = np.mean(pkts_dst)
            temp_dict["dest_std"] = np.std(pkts_dst)
            temp_dict["size_pkt_mean"] = np.mean(window["msglen"])
            temp_dict["size_pkt_std"] = np.std(window["msglen"])

            # TODO finish features extraction
            # add average value change of seq field of packets with the same value of src field.

            # print(temp_dict)
            dfs.append(pd.DataFrame(temp_dict, index=[0]))

    result_df = pd.concat(dfs)
    normalized_df = (result_df - result_df.min()) / (result_df.max() - result_df.min())
    return normalized_df


def plot_pca(df: pd.DataFrame):
    result_np = df.to_numpy()
    pca = PCA(2)  # target dimension
    projected = pca.fit_transform(result_np)
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title("PCA of Dataset " + target_experiment)
    plt.suptitle(f"time windows: {time_window} ms")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    experiment_I = "experiment_I_rpi.csv"
    experiment_II = "experiment_II_lpn.csv"
    target_experiment = experiment_I
    path = "data/" + target_experiment
    time_window = 1000000

    print("Started preprocessing")
    res_df = preprocessing_phase(source_path=path, t_window=time_window)
    print("Finished preprocessing")

    plot_pca(res_df)

# print(df["sent time"])

# df["sent time"] = (df["sent time"] - min_time)
