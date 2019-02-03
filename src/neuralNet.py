# import numpy as np
# import math
# import copy
# import matplotlib.pyplot as plt
# import time


class neuralNet():


    def __init__(self, d, hidden_dims, m, n, eta=3e-4, regularize=None, fixed=False):
        import numpy as np

        self.inputDim = d #inputDim
        self.hiddenDim = hidden_dims # a tuple of hidden units
        self.outputDim = m #outputDim
        self.regularize = regularize # lambda value
        self.learningRate = eta
        self.numData = n
        self.batchErrorGradients = []
        #may use xavier init - maybe explore this later.
        # Initial weights and biases
        if fixed:
            self.W_1 = w1_fixed
            self.W_2 = w2_fixed
            self.W_3 = w3_fixed
        else:
            import numpy as np
            self.W_1 = np.random.uniform(-1/np.sqrt(d), 1/np.sqrt(d),
                                         self.hiddenDim[0]*d).reshape(self.hiddenDim[0], d)
            self.W_2 = np.random.uniform(-1/np.sqrt(self.hiddenDim[0]), 1/np.sqrt(self.hiddenDim[0]),
                                         self.hiddenDim[1]*self.hiddenDim[0]).reshape(self.hiddenDim[1],
                                                                                      self.hiddenDim[0])
            self.W_3 = np.random.uniform(-1/np.sqrt(self.hiddenDim[1]), 1/np.sqrt(self.hiddenDim[1]),
                                         self.hiddenDim[1]*m).reshape(m, self.hiddenDim[1])


        self.b_1 = np.zeros(self.hiddenDim[0]).reshape(self.hiddenDim[0],)
        self.b_2 = np.zeros(self.hiddenDim[1]).reshape(self.hiddenDim[1],)
        self.b_3 = np.zeros(m).reshape(m,)


    def fprop(self, batchData, mode='matrix'):
        '''
        a switch to work for both matrix and loop
        '''

        # hidden layer 1

        if mode == 'matrix':
            #print('self.b1', self.b_1.shape)
            #print('self.W_1', self.W_1.shape)
            #print('batchData.T', batchData.T.shape)
            stack_b1 = np.array([self.b_1,] * self.numData).T
            #print('stack_b1', stack_b1.shape)
            self.h_a1 = np.dot(self.W_1, batchData.T) + stack_b1
        elif mode == 'loop':
            self.h_a1 = np.dot(self.W_1, batchData.T) + self.b_1


        self.h_s1 = relu(self.h_a1)



        # hidden layer 2
        if mode == 'matrix':
            stack_b2 = np.array([self.b_2,] * self.numData).T
            self.h_a2 = np.dot(self.W_2, self.h_s1) + stack_b2
        elif mode == 'loop':
            self.h_a2 = np.dot(self.W_2, self.h_s1) + self.b_2

        self.h_s2 = relu(self.h_a2)

        #output layer weights
        if mode == 'matrix':
            stack_b3 = np.array([self.b_3,] * self.numData).T
            self.o_a = np.dot(self.W_3, self.h_s2) + stack_b3
        elif mode == 'loop':
            self.o_a = np.dot(self.W_3, self.h_s2) + self.b_3

        # softmax of weights
        if batchData.shape[0] == 1:
            print('using single softmax')
            self.o_s = softmax_single(self.o_a)
        else:
            self.o_s = softmax_multiple(self.o_a)

        # make predication
        if mode == 'loop':
            self.prediction = np.argmax(self.o_s,axis = 0)
        elif mode == 'matrix':
            self.prediction = np.argmax(self.o_s,axis = 0)

    def errorRate(self, y, mode='matrix'):
        '''
        negative log
        -logO_s(x)
        HAD the indexing problem for matrix mode
        '''

        if mode == 'loop':
            negLog = -self.o_a[np.argmax(y)] + np.log(np.sum(np.exp(self.o_a), axis=0))

        elif mode == 'matrix':
            negLog = []
            for i in range(y.shape[1]):
                error_at_point = -self.o_a[np.argmax(y[:,i])][i] + np.log(np.sum(np.exp(self.o_a), axis=0))[i]
                negLog.append(error_at_point)
            negLog = np.array(negLog)
            negLog = np.mean(negLog)

        return negLog

    def classErr(self, target, predicted):
        '''
        not class dependent
        target must NOT be in one hot
        '''
        cnt = 0

        for i in range(target.shape[0]):
            if target[i] != predicted [i]:
                cnt +=1
        return float(cnt) / target.shape[0]

    def bpropLoop(self, batchData, batchTarget):
        '''
        dimensions:
        o_s: m x1
        grad_oa : m x 1
        hs: dh x 1
        grad_w2: m x dh
        grad_oa: m x n
        grad_b2: m x n
        grad_oa: m x n
        W(2): m x dh
        grad_hs: dh x n
        grad_oa: m x n
        grad_ha: dh x n
        x : n x d
        grad_W1: dh x d
        grad_ha: dh x n
        grad_b1: dh x n
        '''

        self.grad_oa = self.o_s - batchTarget
        # hidden layer 3
        self.grad_W3 = np.outer(self.grad_oa, self.h_s2.T)
        self.grad_b3 = self.grad_oa
        self.grad_hs2 = np.dot(self.W_3.T , self.grad_oa)
        h_a_stack2 = np.where(self.h_a2 > 0, 1, 0)
        self.grad_ha2 = np.multiply(self.grad_hs2, h_a_stack2)

        # hidden layer 2
        self.grad_W2 = np.outer(self.grad_ha2, self.h_s1.T)
        self.grad_b2 = self.grad_ha2
        self.grad_hs1 = np.dot(self.W_2.T , self.grad_ha2)
        h_a_stack1 = np.where(self.h_a1 > 0, 1, 0)
        self.grad_ha1 = np.multiply(self.grad_hs1, h_a_stack1)

        # hidden layer 1
        self.grad_W1 = np.outer(self.grad_ha1, batchData)
        self.grad_b1 = self.grad_ha1


    def bprop_matrix(self, batchData, batchTarget):
        '''
        backprop using matrix only
        '''

        self.grad_oa = self.o_s - batchTarget

        self.grad_W3 = np.matmul(self.grad_oa, self.h_s2.T)/batchData.shape[0] #!
        self.grad_b3 = np.sum(self.grad_oa, axis=1)/batchData.shape[0] #!
        self.grad_hs2 = np.matmul(self.W_3.T , self.grad_oa)
        self.grad_ha2 = np.multiply(self.grad_hs2, np.where(self.h_a2 > 0, 1.0, 0.0))


        self.grad_W2 = np.matmul(self.grad_ha2, self.h_s1.T)/batchData.shape[0] #!
        self.grad_b2 = np.sum(self.grad_ha2, axis =1)/batchData.shape[0]
        self.grad_hs1 = np.matmul(self.W_2.T, self.grad_ha2)
        self.grad_ha1 = np.multiply(self.grad_hs1, np.where(self.h_a1 > 0, 1.0, 0.0))

        self.grad_W1 = np.matmul(self.grad_ha1, batchData)/batchData.shape[0] #!
        self.grad_b1 = np.sum(self.grad_ha1, axis=1)/batchData.shape[0] #!

    def bprop(self, batchData, batchTarget, mode='matrix'):
        '''
        batchTarget already in one-hot format

        NOT working for a single point

        '''

        #batch target must be m by n
        self.grad_oa = self.o_s - batchTarget
        i = 0
        self.grad_W2 = [np.outer(self.grad_oa[:,i], self.h_s[:,i].T) for i in range(batchData.shape[0])]
        self.grad_b2 = self.grad_oa
        self.grad_hs = np.dot(self.W_2.T , self.grad_oa)
        # Check this (dim mismatch maybe)
        h_a_stack = np.where(self.h_a > 0, 1, 0)
        self.grad_ha = np.multiply(self.grad_hs, h_a_stack)
        #self.grad_W1 = [np.outer(self.grad_ha[:,i], batchData[i]) for i in range(self.numData)]
        self.grad_W1 = [np.outer(self.grad_ha[:,i], batchData[i]) for i in range(batchData.shape[0])]
        # temporary hack for grad_W
        self.grad_b1 = self.grad_ha


        if mode == 'matrix':
            '''
            must avg,
            1 pt would return a list of MAT/np array, not a NP array
            '''
            self.grad_W2 = np.average(np.array(self.grad_W2), axis=0)
            self.grad_b2 = np.average(np.array(self.grad_b2), axis=1)

            self.grad_W1 = np.average(np.array(self.grad_W1), axis=0)
            self.grad_b1 = np.average(np.array(self.grad_b1), axis=1)



    def updateParams(self):
        if self.regularize:
            self.W_1 -= (self.regularize[0] * np.sign(self.W_1) + 2 * self.regularize[1] * self.W_1) * self.learningRate
            self.W_2 -= (self.regularize[2] * np.sign(self.W_2) + 2 * self.regularize[3] * self.W_2) * self.learningRate
            self.W_3 -= (self.regularize[4] * np.sign(self.W_3) + 2 * self.regularize[5] * self.W_3) * self.learningRate


        self.W_1 -= self.grad_W1 * self.learningRate
        self.W_2 -= self.grad_W2 * self.learningRate
        self.W_3 -= self.grad_W3 * self.learningRate

        self.b_1 -= self.grad_b1 * self.learningRate
        self.b_2 -= self.grad_b2 * self.learningRate
        self.b_3 -= self.grad_b3 * self.learningRate




    def gradDescentLoop(self, batchData, batchTarget, K):
        # Call each example in the data (over the minibatches) in a loop
        grad_W3, grad_b3, grad_W2, grad_b2, grad_W1, grad_b1 = [], [], [], [], [], []
        predBatch = []
        for i in range(K):
            self.fprop(batchData[i], mode='loop') #batchTarget[:,i]
            self.bpropLoop(batchData[i],np.array(batchTarget[:,i]))
            predBatch.append(self.prediction)
            grad_W3.append(self.grad_W3)
            grad_b3.append(self.grad_b3)
            grad_W2.append(self.grad_W2)
            grad_b2.append(self.grad_b2)
            grad_W1.append(self.grad_W1)
            grad_b1.append(self.grad_b1)

        self.grad_W3 = np.mean(np.array(grad_W3), axis=0) #! array
        self.grad_b3 = np.mean(np.array(grad_b3), axis=0)
        self.grad_W2 = np.mean(np.array(grad_W2), axis=0) #! array
        self.grad_b2 = np.mean(np.array(grad_b2), axis=0)
        self.grad_W1 = np.mean(np.array(grad_W1), axis=0) #! array
        self.grad_b1 = np.mean(np.array(grad_b1), axis=0)

        # Update params
        #self.updateParams()

    def fpropLoop(self, batchData, K):
        '''
        unlike the above def gradDescentLoop(self, batchData, batchTarget, K)
        this function only runs batchData (this is usually in test phase)
        through the forward prop, without calculating any gradient update rule.

        Use to get predictions

        batchData: more like test/val data
        K: ALWAYS == batchData.shape[0]

        '''
        predBatch = []
        for i in range(K):
            self.fprop(batchData[i], mode='loop') #batchTarget[:,i]
            predBatch.append(self.prediction)
        self.predBatch = np.array(predBatch)

    def gradDescentMat(self, batchData, batchTarget):
        # Feed the entire data matrix in as input
        self.fprop(batchData)
        self.bprop_matrix(batchData, batchTarget)
