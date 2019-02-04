from src.algo.error_func import show_error, classErr
from src.algo.sampler import BatchSampler
import numpy as np


def train_matrix(nn, data, target, K, num_epoch, save_directory, fixed=False, valid=None, test=None):
    # Get minibatch
    batchSampler = BatchSampler(data, target, K)
    numBatch = data.shape[0] // K

    print("number of batch in train matrix", numBatch)
    for n in range(num_epoch):
        for i in range(numBatch):
            # Do descent and update params - this is one epoch

            if fixed:
                batchData, batchTarget = batchSampler.get_batch(K)
            elif not fixed:
                batchData, batchTarget = batchSampler.get_batch()
            nn.numData = K
            nn.gradDescentMat(batchData, batchTarget.T)

            nn.updateParams()
        if n % 10 == 0:
            print(':)')
            nn.fprop(batchData, mode = 'matrix')
            pred = np.argmax(nn.o_s, axis = 0)
            print("Cross-entropy loss at the end of epoch {}: {}".format(n, nn.errorRate(batchTarget.T, mode = 'matrix')))
            print("classification error at the end of epoch {}: {}".format(n,
                                                    classErr(np.argmax(batchTarget, axis = 1), pred)))
        if valid:
            nn.numData = valid[0].shape[0]
            show_error(save_directory, nn, n, [data, target], valid, test)
    print("End of train matrix process.")