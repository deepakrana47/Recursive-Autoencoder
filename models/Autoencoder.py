import numpy as np

class autoencoder:

    def __init__(self, hidden_activation, hidden_deactivation, output_activation, output_deactivation):
        self.hidactivation = hidden_activation
        self.hiddeactivation = hidden_deactivation
        self.outactivation = output_activation
        self.outdeactivation = output_deactivation
        pass

    def encoding(self, x, we, be):
        ###### Encoding
        to = np.dot(we, x) + be
        h = self.hidactivation(to)
        return h, to

    def decoding(self, h, wd, bd):
        ###### Decoding
        to_ = np.dot(wd, h) + bd
        y = self.outactivation(to_)
        return y, to_

    def backward_pass(self, grad, we, wd, to, to_):
        ograd = grad * self.outdeactivation(to_)
        hgrad = np.dot(wd.T, ograd) * self.hiddeactivation(to)
        lgrad = np.dot(we.T, hgrad)
        return hgrad, ograd, lgrad