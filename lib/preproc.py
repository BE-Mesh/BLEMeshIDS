"""preproc.py
Handle preprocessing
"""

import pandas as pd
import numpy as np
import os
import logging

LABEL_LST = ['legit', 'black_hole', 'gray_hole']


DATA_COLS = ['timestamp',
             'N',
             'role',
             'seq',
             'net_idx',
             'app_idx',
             'src',
             'dst',
             'rssi',
             'ttl',
             'buf_len',
             'zeroes']


def read_data(file: str) -> pd.DataFrame:
    """
    Read a file in CSV format and returns the associated DataFrame
    :param file: string pointing to the target file
    :return: the file as a DataFrame
    """
    return pd.read_csv(file, names=DATA_COLS, engine='python')


def generate_time_windows(data: pd.DataFrame, window_len: int) -> [pd.DataFrame]:
    """generate_time_windows
    Generates the list of time windows of dimension window_len.
    :type window_len: int
    :type data: pd.DataFrame
    :param data: The data object in form of DataFrame
    :param window_len: length of each time window, must be expressed in microseconds
    :return: List of DataFrame(s) divided by time windows
    """

    min_time = data['timestamp'].min()
    # Alter the time_window column in the dataset by applying a discrete transform that subdivides each window with a
    # different index
    data['time_window'] = np.floor((data['timestamp'].astype(np.int64) - min_time) / window_len)
    # Time windows list
    dfs = []
    for i in range(int(data['time_window'].max())):
        window = data.loc[data['time_window'] == i]
        if not window.empty:
            dfs.append(window)
    return dfs


def compute_features(window: pd.DataFrame) -> np.array:
    """
    Take a time window as input and output the temporal analysis (TA) features as a numpy array.
    The following features are extracted:
    (size, rssi_mean, rssi_std, ttl_mean, ttl_std, src_mean, src_std, dst_mean, dst_std, size_pkt_mean, size_pkt_std)
    :param window: DataFrame representing a time window
    :return: Numpy array containing extracted features
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


def preproc_data(file: str, window_len: int, verbose=False) -> np.array:
    df = read_data(file)
    twin_lst = generate_time_windows(df, window_len)
    preproc_lst = [compute_features(x) for x in twin_lst]
    return np.array(preproc_lst)


def generate_dataset(file: str, label: int, window_len=int(1e6)) -> (np.array, np.array):
    X = preproc_data(file, window_len)
    y = np.array([label for _ in range(X.shape[0])])
    return X, y


def load_dataset(file: str, label: int) -> (np.array, np.array):
    X = np.loadtxt(file)
    y = np.array([label for _ in range(X.shape[0])])
    return X, y


def load_dataset_folder(path: str, labels: [str] = None):
    X_lst = []
    y_lst = []
    if labels is None:
        labels = ['legit', 'black_hole', 'gray_hole']
    for i, label in enumerate(labels):
        data_path = os.path.join(path, label)
        for fname in os.listdir(data_path):
            if fname.endswith('.csv'):
                X, y = load_dataset(
                    os.path.join(data_path, fname), i)
                X_lst.append(X)
                y_lst.append(y)
    return X_lst, y_lst


def load(path: str, labels: [str] = None) -> ([np.array], [np.array]):
    X_lst = []
    y_lst = []
    if labels is None:
        logging.warning('No labels passed. Using default')
        labels = LABEL_LST
    for i, label in enumerate(labels):
        data_path = os.path.join(path, label)
        for fname in os.listdir(data_path):
            X, y = load_dataset(os.path.join(data_path, fname), i)
            X_lst.append(X)
            y_lst.append(y)
    return X_lst, y_lst


if __name__ == '__main__':
    # X, y = generate_dataset('test.csv', label=0, window_len=int(1e6))
    # print(X.shape, y.shape)
    # TODO(FIX new file system)
    # X_lst, y_lst = load_dataset_folder('../data')
    exit(0)
