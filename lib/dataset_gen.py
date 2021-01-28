"""dataset_gen
Dataset generator script"""

import numpy as np
from os.path import join
import os
from lib.preproc import preproc_data

DATA_LEGIT_PATH = 'data/legit.csv'
DATA_BH_PATH = 'data/black_hole.csv'
DATA_GH_PATH = 'data/grey_hole.csv'
WSIZE = int(1e6)

DATA_LST = [DATA_LEGIT_PATH, DATA_BH_PATH, DATA_GH_PATH]
LABEL_LST = ['legit', 'black_hole', 'grey_hole']

"""DATA_LST = [DATA_BH_PATH, DATA_GH_PATH]
LABEL_LST = ['black_hole', 'grey_hole']"""

OUTPUT_DATASET_PATH = 'data/'

"""
def generate_dataset(window_size: int = WSIZE):
    for i, data_path in enumerate(DATA_LST):
        x = preproc_data(data_path, window_size)
        base_path = join(OUTPUT_DATASET_PATH, LABEL_LST[i])
        np.savetxt(join(base_path, '0.csv'), x)
"""


def generate_dataset(raw_path: str, proc_path: str, window_size: int = WSIZE):
    """
    Generates the set of preprocessed datasets for each type of experiment.
    Input data should be stored in raw_path whereas Output data is saved in proc_path.
    :param raw_path:
    :param proc_path:
    :param window_size:
    :return:
    """
    # Verify that legit, gray_hole and black_hole folders are present both in raw_path and proc_path
    raw_dir_lst = os.listdir(raw_path)
    proc_dir_lst = os.listdir(proc_path)
    assert raw_dir_lst == proc_dir_lst

    # Iterate on the three folders at if possible, start generating the dataset.
    for label in LABEL_LST:
        raw_dir = os.path.join(raw_path, label)
        proc_dir = os.path.join(proc_path, label)
        data_lst = os.listdir(raw_dir)
        for i in range(len(data_lst)):
            data_path = os.path.join(raw_dir, data_lst[i])
            out_data = preproc_data(data_path, window_size, True)
            np.savetxt(join(proc_dir, f'{i+1}.csv'), out_data)


if __name__ == '__main__':
    generate_dataset('data/raw', 'data/proc')
    exit(0)
