import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential


class ANN:
    def __init__(
        self, name, input_dim, output_dim, layers, activation, loss, optimizer
    ):
        model = Sequential()

        first = True
        for num_nodes in layers:
            if first:
                model.add(Dense(num_nodes, input_dim=input_dim, activation=activation))
                first = False
            else:
                model.add(Dense(num_nodes))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.name       = name
        self.model      = model

    # Supervised learning
    def train(self, X, y, batch_size):
        return self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

    def predict(self, state):
        try:
            reshaped = np.asarray(state).reshape((1, self.input_dim))
        except ValueError:
            reshaped = state
        prediction = self.model.predict(reshaped)
        return prediction

    # Weights management
    def set(self, weights):
        self.model.set_weights(weights)

    def get(self):
        return self.model.get_weights()

    def dump(self):
        self.model.save_weights('.weights/{}.h5'.format(self.name), overwrite=True)

    def load(self):
        try:
            self.model.load_weights('.weights/{}.h5'.format(self.name))
        except IOError:
            self.model.load_weights('.weights/{}.h5'.format(self.name.replace('ann', 'target_ann')))
