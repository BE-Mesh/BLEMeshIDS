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
    """
    # TODO
    return []

def compute_features(data: pd.DataFrame) -> np.array:
    """compute_features
    Take a time window as input and output the temporal analysis (TA) features as a numpy array
    """
    # TODO
    return data.to_numpy()

def preproc_data(file: str, window_len: int) -> np.array:
    df = read_data(file)
    twin_lst = generate_time_windows(df, window_len)
    preproc_lst = [compute_features(x) for x in twin_lst]
    return np.array(preproc_lst)

if __name__ == '__main__':
    data = read_data('test.csv')

    data = preproc_data('....csv')