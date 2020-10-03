"""preproc.py
Handle preprocessing
"""

import pandas as pd
import numpy as np
import os

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

    min_time = data['timestamp'].min()
    data['time_window'] = np.floor((data['timestamp'].astype(np.int64) - min_time) / window_len)
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

    unique, counts = np.unique(window['dst'], return_counts=True)
    pkts_dst = list(dict(zip(unique, counts)).values())  # packets per dest
    temp_dict['dst_mean'] = np.mean(pkts_dst)
    temp_dict['dst_std'] = np.std(pkts_dst)
    temp_dict['size_pkt_mean'] = np.mean(window['buf_len'])
    temp_dict['size_pkt_std'] = np.std(window['buf_len'])
    return np.array(list(temp_dict.values()))


def preproc_data(file: str, window_len: int) -> np.array:
    print('reading file...', end='')
    df = read_data(file)
    print('done.')
    print('generating time windows...', end='')
    twin_lst = generate_time_windows(df, window_len)
    print('done.')
    print('computing features...', end='')
    preproc_lst = [compute_features(x) for x in twin_lst]
    print('done.')
    return np.array(preproc_lst)


def generate_dataset(file: str, label: int, window_len=int(1e6)) -> (np.array, np.array):
    X = preproc_data(file, window_len)
    y = np.array([label for _ in range(X.shape[0])])
    return X, y


def load_dataset(file: str, label: int) -> (np.array, np.array):
    X = np.loadtxt(file)
    y = np.array([label for _ in range(X.shape[0])])
    return X, y

def load_dataset_folder(path: str, labels: [str]=None):
    X_lst = []
    y_lst = []
    if labels is None:
        labels = ['legit', 'black_hole', 'grey_hole']
    for i, label in enumerate(labels):
        data_path = os.path.join(path, label)
        for fname in os.listdir(data_path):
            if fname.endswith('.csv'):
                X, y = load_dataset(
                    os.path.join(data_path, fname), i)
                X_lst.append(X)
                y_lst.append(y)
    return X_lst, y_lst


if __name__ == '__main__':
    #X, y = generate_dataset('test.csv', label=0, window_len=int(1e6))
    #print(X.shape, y.shape)
    X_lst, y_lst = load_dataset_folder('../data')
