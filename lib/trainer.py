"""trainer.py

"""

import numpy as np
import tensorflow as tf
from lib.preproc import load_dataset

DATA_LEGIT_PATH = ''
DATA_BH_PATH = ''
WSIZE = int(1e5)

LABELS = {'legit': 0, 'black_hole': 1, 'grey_hole': 2}

if __name__ == '__main__':
    x_l, y_l = load_dataset(DATA_LEGIT_PATH, LABELS['legit'], WSIZE)
    exit(0)
