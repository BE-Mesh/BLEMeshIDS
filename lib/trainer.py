"""trainer.py

"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize

from lib.preproc import load_dataset_folder
from sklearn.utils import shuffle, compute_class_weight
from sklearn.model_selection import train_test_split
import datetime

DATA_LEGIT_PATH = ''
DATA_BH_PATH = ''
WSIZE = int(2e6)
BATCH_SIZE = 128
EPOCHS = 3000
NUM_CLASSES = 3

LABELS = {'legit': 0, 'black_hole': 1, 'grey_hole': 2}

SAVE_LOGS_DIR = 'logs/'
DATASET_DIR = 'data/'


def stack_data(X_lst: [np.array], y_lst: [np.array], onehot=True, num_classes=2):
    X = np.concatenate(X_lst, axis=0)
    y = np.concatenate(y_lst, axis=0)
    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)
    if onehot:
        y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    return X, y


def generate_model_01(batch_size, num_classes=2):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, input_shape=(batch_size, 11), activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def train():
    X_lst, y_lst = load_dataset_folder(DATASET_DIR)
    X, y = stack_data(X_lst, y_lst, onehot=True, num_classes=NUM_CLASSES)
    normalized_X = normalize(X, axis=1, norm='l2')
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.33, random_state=42)
    # Convert to tf.data.Dataset
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_data = train_data.batch(BATCH_SIZE).repeat().cache()
    test_data = test_data.batch(BATCH_SIZE)

    model = generate_model_01(BATCH_SIZE, num_classes=NUM_CLASSES)

    # Compute class weights
    class_weights = compute_class_weight('balanced', np.unique(np.argmax(y_train, axis=1)), np.argmax(y_train, axis=1))
    print(class_weights)

    # TensorBoard callback
    log_dir = SAVE_LOGS_DIR + 'fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    cb_tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    training_history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data, steps_per_epoch=15,
                                 class_weight={0: class_weights[0], 1: class_weights[1], 2: class_weights[2]},
                                 callbacks=[cb_tb])

    # Store training history
    np.savetxt(SAVE_LOGS_DIR + 'acc.txt', training_history.history['acc'])
    np.savetxt(SAVE_LOGS_DIR + 'val_acc.txt', training_history.history['val_acc'])
    np.savetxt(SAVE_LOGS_DIR + 'loss.txt', training_history.history['loss'])
    np.savetxt(SAVE_LOGS_DIR + 'val_loss.txt', training_history.history['val_loss'])

    model.save_weights(SAVE_LOGS_DIR + 'weights.h5')


if __name__ == '__main__':
    train()
    exit(0)
