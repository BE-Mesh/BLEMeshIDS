from tensorflow import keras


def create_binary_model():
    return keras.Sequential([
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])


def create_multiclass_model():
    return 0


def create_model(kind: str):
    return create_binary_model() if kind == "binary" else create_multiclass_model()
