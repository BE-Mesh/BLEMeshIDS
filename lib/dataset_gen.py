"""dataset_gen
Dataset generator script"""

import numpy as np
from os.path import join
from lib.preproc import preproc_data

DATA_LEGIT_PATH = 'legit.csv'
DATA_BH_PATH = 'bh_2.csv'
DATA_GH_PATH = 'test_gh.csv'
WSIZE = int(1e6)

DATA_LST = [DATA_LEGIT_PATH, DATA_BH_PATH, DATA_GH_PATH]
LABEL_LST = ['legit', 'black_hole', 'grey_hole']

"""DATA_LST = [DATA_BH_PATH, DATA_GH_PATH]
LABEL_LST = ['black_hole', 'grey_hole']"""

OUTPUT_DATASET_PATH = '../data/'

if __name__ == '__main__':
    for i, data_path in enumerate(DATA_LST):
        x = preproc_data(data_path, WSIZE)
        base_path = join(OUTPUT_DATASET_PATH, LABEL_LST[i])
        np.savetxt(join(base_path, '0.csv'), x)
    exit(0)