import pandas as pd
import numpy as np


def data_loader(path: str, label: int):
    df = pd.read_csv(path)
    x = df.to_numpy()
    shaped = x.shape
    # print(shaped)
    # print(shaped[0])
    y = [label for i in range(shaped[0])]
    # print(y)
    # print(len(y))


if __name__ == '__main__':
    experiment_I = "experiment_I_rpi.csv"
    experiment_II = "experiment_II_lpn.csv"
    target_experiment = experiment_I
    processed_path = "../data/preprocessed_" + target_experiment
    data_loader(processed_path, 0)
