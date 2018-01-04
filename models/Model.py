from abc import abstractmethod, ABCMeta
from util.Utility import mini_batch, unzip
import pickle

class Layer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_pass(self, x, extra=None):
        pass

    @abstractmethod
    def backward_pass(self, grad, extra=None):
        pass

    @abstractmethod
    def calc_weight_update(self, x, grad, extra=None):
        pass

    @abstractmethod
    def update_weights(self):
        pass

class model:

    def __init__(self):
        self.layers = []
        pass

    def add_layer(self, layer):
        self.layers.append(layer)
        pass

    def train(self, x, y, epoch, batch_size):

        for i in range(epoch):

            batches = mini_batch(zip(x,y), len(x), batch_size)
            for batch in batches:

                for x,y in batch:

                    # Forward pass
                    self.output=[x]
                    for i in range(len(self.layers)):
                        self.output.append(self.layers[i].forward_pass(self.output[i]))

                    # backward_pass
                    self.grad = {len(self.layers):self.layers[-1].error(self.output[-1], y)}
                    for i in reversed(range(len(self.layers))):
                        self.grad[i]=self.layers[i].backward_pass(self.grad[i+1])

                    # calculating weight update
                    for i in range(len(self.layers)):
                        self.layers[i].calc_weight_update(self.output[i], self.grad[i+1])

                # updating weight
                for i in range(len(self.layers)):
                    self.layers[i].update_weights()

def save_model(obj, fname):
    pickle.dump(obj, open(fname, 'wb'))

def load_model(fname):
    return pickle.load(open(fname, 'rb'))