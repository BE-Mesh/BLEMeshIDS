import json
import random

import numpy as np
from sklearn.model_selection import train_test_split

from DataLoader.data_loader import data_loader
from DataParser.data_parser import preprocessing_phase
import tensorflow as tf


def train(experiments: dict, labels_values: dict, history_path: str, model, model_path: str, model_weights_path: str):
    images: np.ndarray
    labels = []
    for i, experiment in enumerate(experiments.keys()):
        path = "data/" + experiment
        time_window = 1000000

        print(f'Started preprocessing for "{experiment}"')
        res_df = preprocessing_phase(source_path=path, t_window=time_window)
        print("Finished preprocessing")

        # print(res_df.head())
        processed_path = "data/preprocessed_" + experiment
        res_df.to_csv(processed_path, index=False)

        x, y = data_loader(processed_path, labels_values[experiments[experiment]])
        images = x if i == 0 else np.concatenate((images, x))
        labels += y

    labels = np.array(labels)
    c = list(zip(images, labels))
    random.shuffle(c)
    x, y = zip(*c)
    x = np.array(x)
    y = np.array(y)

    # for scikit learn
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # for tensorflow
    training_set = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_set = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    training_set = training_set.batch(32).cache().repeat()
    test_set = test_set.batch(32)

    cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=True)

    history_model = model.fit(training_set, epochs=600, steps_per_epoch=15, validation_data=test_set, callbacks=[cb_es])

    model.save(model_path)
    model.save_weights(model_weights_path)
    json.dump(history_model.history, open(history_path, 'w'))
