"""dataset_gen
Dataset generator script"""

import numpy as np
from os.path import join
from lib.preproc import preproc_data

DATA_LEGIT_PATH = '../data/legit.csv'
DATA_BH_PATH = '../data/black_hole.csv'
DATA_GH_PATH = '../data/grey_hole.csv'
WSIZE = int(1e6)

DATA_LST = [DATA_LEGIT_PATH, DATA_BH_PATH, DATA_GH_PATH]
LABEL_LST = ['legit', 'black_hole', 'grey_hole']

"""DATA_LST = [DATA_BH_PATH, DATA_GH_PATH]
LABEL_LST = ['black_hole', 'grey_hole']"""

OUTPUT_DATASET_PATH = '../data/'


def generate_dataset(window_size: int = WSIZE):
    for i, data_path in enumerate(DATA_LST):
        x = preproc_data(data_path, window_size)
        base_path = join(OUTPUT_DATASET_PATH, LABEL_LST[i])
        np.savetxt(join(base_path, '0.csv'), x)


if __name__ == '__main__':
    generate_dataset()
    exit(0)
