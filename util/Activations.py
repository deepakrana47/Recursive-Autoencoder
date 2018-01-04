import numpy as np

def sigmoid(x):
    return np.power(1+np.exp(-x), -1)

def dsigmoid(x):
    t=sigmoid(x)
    return (1-t)*t

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return  1-np.square(np.tanh(x))

def relu(x):
    return np.where(x<0,0,x)

def drelu(x):
    return np.where(x>0,1,0.01)

def etanh(x):
    return np.e * np.tanh( x / np.e)

def detanh(x):
    return  1-np.square(np.tanh(x)/np.e)

def elu(x):
    return np.where(x>0,x,np.exp(x)-1)

def delu(x):
    return np.where(x>0,1.0,np.exp(x))