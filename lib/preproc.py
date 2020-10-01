"""preproc.py
Handle preprocessing
"""

import pandas as pd
import numpy as np

DATA_COLS = ['timestamp', 'role', 'seq', 'net_idx', 'app_idx', 'src', 'dst', 'rssi', 'ttl', 'buf_len']


def read_data(file: str) -> pd.DataFrame:
    return pd.read_csv(file, names=DATA_COLS, engine='python')


def generate_time_windows(data: pd.DataFrame, window_len: int) -> [pd.DataFrame]:
    """generate_time_windows
    Generates the list of time windows of dimension window_len.
    :type window_len: int
    :type data: pd.DataFrame
    :param data:
    :param window_len: length of each time window, must be expressed in microseconds
    :return:
    """
    min_time = data['time_window'].min()
    data['time_window'] = np.floor((data['timestamp'] - min_time) / window_len)
    dfs = []
    for i in range(int(data['time_window'].max())):
        window = data.loc[data['time_window'] == i]
        if not window.empty:
            dfs.append(window)

    return dfs


def compute_features(window: pd.DataFrame) -> np.array:
    """compute_features
    Take a time window as input and output the temporal analysis (TA) features as a numpy array
    """
    temp_dict = {'size': window.size, 'rssi_mean': np.mean(window['rssi']), 'rssi_std': np.std(window['rssi']),
                 'ttl_mean': np.mean(window['ttl']), 'ttl_std': np.std(window['ttl'])}
    unique, counts = np.unique(window['src'], return_counts=True)
    pkts_src = list(dict(zip(unique, counts)).values())  # packets per source
    temp_dict['src_mean'] = np.mean(pkts_src)
    temp_dict['src_std'] = np.std(pkts_src)

    # can be added average value change of seq field of packets with the same value of src field.
    # temp_dict['seq_mean'] = np.mean(window['seq'])

    unique, counts = np.unique(window['dest'], return_counts=True)
    pkts_dst = list(dict(zip(unique, counts)).values())  # packets per dest
    temp_dict['dest_mean'] = np.mean(pkts_dst)
    temp_dict['dest_std'] = np.std(pkts_dst)
    temp_dict['size_pkt_mean'] = np.mean(window['buf_len'])
    temp_dict['size_pkt_std'] = np.std(window['buf_len'])

    return np.array(list(temp_dict.values()))


def preproc_data(file: str, window_len: int) -> np.array:
    df = read_data(file)
    twin_lst = generate_time_windows(df, window_len)
    preproc_lst = [compute_features(x) for x in twin_lst]
    return np.array(preproc_lst)


def load_dataset(file: str, label: int, window_len=int(1e3)) -> (np.array, np.array):
    X = preproc_data(file, window_len)
    y = np.array([label for _ in range(X.shape[0])])
    return X, y


if __name__ == '__main__':
    X, y = load_dataset('test.csv', label=0, window_len=1000)
    print(X.shape, y.shape)
