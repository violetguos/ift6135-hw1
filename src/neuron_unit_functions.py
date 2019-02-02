import numpy as np
def classErr(target, predicted):
    '''
    not class dependent
    target must NOT be in one hot
    '''
    cnt = 0
    #print("in class Err")
    #print("target \n", target.shape[0])
    #print("predicted \n", predicted.shape[0])
    #print("target range \n", np.max(target))
    #print("predicted range \n", np.max(predicted))
    for i in range(target.shape[0]):
        if target[i] != predicted [i]:
            cnt +=1
    return float(cnt) / target.shape[0]

# Our own activation functions

def relu(pre_activation):
    '''
    preactivation is a vector
    '''
    relu_output = np.zeros(pre_activation.shape)
    relu_flat = relu_output.flatten()
    for i, neuron in enumerate(pre_activation.flatten()):
        if neuron > 0:
            relu_flat[i] = neuron
    relu_output = relu_flat.reshape(pre_activation.shape)
    return relu_output

def softmax_single(pre_activation):
    '''
    Numerically stable because subtracting the max value makes bit overflow impossible,
    we will only have non-positive values in the vector
    '''
    exps = np.exp(pre_activation - np.max(pre_activation))
    return exps / np.sum(exps)

def softmax_multiple(pre_activation):
    '''
    Numerically stable because subtracting the max value makes bit overflow impossible,
    we will only have non-positive values in the vector
    '''
    exps = np.exp(pre_activation - np.max(pre_activation, axis = 0))
    return exps / np.sum(exps, axis = 0)
