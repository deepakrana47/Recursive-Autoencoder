import numpy as np

class Optimization:

    def __init__(self, optimization_variable, method='rmsprop', learning_rate=0.001, regularization=0.0001, options=None):
        self.neta = learning_rate
        self.opt_var = optimization_variable
        self.regu = regularization
        self.method = method
        if method == 'adam':
            self.update = self.adam
            if options:
                self.b1, self.b2, self.e = options['b1'], options['b2'], options['e']
            else:
                self.b1, self.b2, self.e = 0.9, 0.99, 1e-8

        elif method == 'rmsprop':
            self.update = self.rmsprop
            if options:
                self.b1, self.e = options['b1'], options['e']
            else:
                self.b1, self.e = 0.9, 1e-8

        elif method == 'sgd':
            self.update = self.sgd

        else :
            print "Optimization method name is not correct !!!"
            exit()

    def adam(self, w, dw, b, db, extra=None):
        m, v = self.opt_var.getm(extra), self.opt_var.getv(extra)
        m = (self.b1 * m) + ((1. - self.b1) * dw)
        v = (self.b2 * v) + ((1. - self.b2) * np.square(dw))
        m_h = m / (1. - self.b1)
        v_h = v / (1. - self.b2)
        # w -= neta * (m_h/(np.sqrt(v_h) + e) + regu * w)
        w -= self.neta * m_h / (np.sqrt(v_h) + self.e)
        self.opt_var.setm(m, extra), self.opt_var.setv(v, extra)
        b -= self.neta * db
        return w, b

    def rmsprop(self, w, dw, b, db, extra=None):
        m = self.opt_var.getm(extra)
        m = self.b1*m + (1 - self.b1)*np.square(dw)
        w -= self.neta * np.divide(dw, (np.sqrt(m) + self.e))
        self.opt_var.setm(m ,extra)
        b -= self.neta * db
        return w, b

    def sgd(self, w, dw, b, db, extra=None):
        w -= self.neta*(dw)# + regu*(w))
        b -= self.neta * (db)
        return w, b

class Optimization_variable:
    def __init__(self, method, i_size, o_size, model_type, option=None):
        self.g = {}
        if method == 'adam':
            self.init_adam_var( i_size, o_size, model_type, option)
        elif method == 'rmsprop':
            self.init_rmsprop_var( i_size, o_size, model_type, option)

    def init_adam_var(self, i_size, o_size, model_type, option):
        if model_type == 'RAE':
            self.g[0] = {'e': {}, 'd': {}}
            self.g[1] = {'e': {}, 'd': {}}
            wpresent = option['wpresent']
            for i in wpresent:
                self.g[0]['e'][i] = np.zeros((o_size, o_size))
                self.g[0]['d'][i] = np.zeros((o_size, o_size))
                self.g[1]['e'][i] = np.zeros((o_size, o_size))
                self.g[1]['d'][i] = np.zeros((o_size, o_size))

            for i in [0, 0.1]:
                self.g[0]['e'][i] = np.zeros((o_size, i_size))
                self.g[0]['d'][i] = np.zeros((i_size, o_size))
                self.g[1]['e'][i] = np.zeros((o_size, i_size))
                self.g[1]['d'][i] = np.zeros((i_size, o_size))
        else:
            self.g[0] = np.zeros((o_size, i_size))
            self.g[1] = np.zeros((o_size, i_size))
        return

    def init_rmsprop_var(self, i_size, o_size, model_type=None, option=None):

        if model_type == 'RAE':
            self.g[0] = {'e': {}, 'd': {}}
            wpresent = option['wpresent']
            for i in wpresent:
                self.g[0]['e'][i] = np.zeros((o_size, o_size))
                self.g[0]['d'][i] = np.zeros((o_size, o_size))

            for i in [0, 0.1]:
                self.g[0]['e'][i] = np.zeros((o_size, i_size))
                self.g[0]['d'][i] = np.zeros((i_size, o_size))
        else:
            self.g[0] = np.zeros((o_size, i_size))
        return

    def getm(self, extra):
        if extra:
            return self.g[0][extra[0]][extra[1]]
        else:
            return self.g[0]

    def getv(self, extra):
        if extra:
            return self.g[1][extra[0]][extra[1]]
        else:
            return self.g[1]

    def setm(self, m, extra):
        if extra:
            self.g[0][extra[0]][extra[1]] = m
        else:
            self.g[0] = m

    def setv(self, v, extra):
        if extra:
            self.g[1][extra[0]][extra[1]] = v
        else:
            self.g[1] = v