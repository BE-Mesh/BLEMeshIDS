"""dataset_gen
Dataset generator script"""

import numpy as np
from os.path import join
import os
from lib.preproc import preproc_data

WSIZE = int(1e6)

LABEL_LST = ['legit', 'black_hole', 'gray_hole']


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
            print(f'Preprocessing {data_path}')
            out_data = preproc_data(data_path, window_size, True)
            np.savetxt(join(proc_dir, f'{i + 1}.csv'), out_data)


if __name__ == '__main__':
    generate_dataset('data/raw', 'data/proc')
    exit(0)
