import pandas as pd
import numpy as np


def preprocessing_phase(source_path: str, t_window: int, clean_bcast: bool = False,
                        normalize: bool = False) -> pd.DataFrame:
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

    if normalize:
        normalized_df = (result_df - result_df.min()) / (result_df.max() - result_df.min())
        normalized_df.reset_index(drop=True, inplace=True)
        return normalized_df

    result_df.reset_index(drop=True, inplace=True)
    return result_df
