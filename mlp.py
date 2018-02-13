import keras, os
from keras.models import Sequential
from keras.layers import Dense,Dropout
class MLP:

    def __init__(self):
        pass

    def make_model(self, input_shape, out_shape):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        # model.add(Dropout(0.5))
        model.add(Dense(254, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(out_shape, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.001), metrics=['accuracy'])
        return model

    def classify(self, inp):
        return self.model.pridect(inp)

    def evaluate(self, x, y):
        return self.model.evaluate(x=x, y=y, batch_size=50, verbose=1)

    def train(self, x, y, modelf=None):
        if os.path.isfile(modelf):
            self.model = keras.models.load_model(modelf)
        else:
            batch_size = 100
            epochs = 150
            self.model = self.make_model(x[0].shape[0], y[0].shape[0])
            self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1)
        return

    def load_model(self, modelf):
        self.model = keras.models.load_model(modelf)

import numpy as np
def keras_preprocess(train, train_label, test, test_label):
    train_label = np.array([[1.0, 0.0] if i else [0.0, 1.0] for i in train_label])
    test_label = np.array([[1.0, 0.0] if i else [0.0, 1.0] for i in test_label])
    return train, train_label, test, test_label
