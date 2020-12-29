from tensorflow import keras
import tensorflow.keras.layers as layers


def cnn(input_shape, num_actions):
    leaky_relu = keras.layers.LeakyReLU(alpha=0.1)
    model = keras.models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation=leaky_relu, input_shape=input_shape,
                            data_format='channels_first'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation=leaky_relu, data_format='channels_first'))
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), activation=leaky_relu, data_format='channels_first'))
    # model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu', data_format='channels_first'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=512, activation=leaky_relu))
    model.add(layers.Dense(units=num_actions, activation='linear'))
    return model
