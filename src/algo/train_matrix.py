import numpy as np
import os

from src.algo.error_func import show_error, classErr
from src.algo.sampler import BatchSampler
from src.utils.cmd_util import file_name_gen
from tqdm import tqdm


def train_matrix(nn, data, target, K, num_epoch, save_directory,
                 valid=None, test=None, return_err=False):
    '''
    'valid' and 'test' must be in a list, as [inputs, targets].
    '''
    # Get minibatch
    batchSampler = BatchSampler(data, target, K)
    numBatch = data.shape[0] // K
    save_freq = int(num_epoch / 10)

    print("number of batch in train matrix", numBatch)
    print('Training...')
    for n in range(num_epoch):
        print('Epoch {}/{}'.format(n, num_epoch))
        # Do descent and update params - this is one epoch
        for i in tqdm(range(numBatch)):
            # Get batch
            batchData, batchTarget = batchSampler.get_batch(K)
            nn.numData = K

            if not valid:
                nn.gradDescentMat(batchData, batchTarget.T)
                nn.updateParams()

                # Save every 10th epoch, and at the end
                if n % save_freq == 0 or n == (num_epoch - 1):
                    print(':) saving')
                    nn.fprop(batchData, mode='matrix')
                    pred = np.argmax(nn.o_s, axis=0)
                    train_err = nn.errorRate(batchTarget.T, mode='matrix')
                    class_err = classErr(np.argmax(batchTarget, axis=1), pred)
                    print("Cross-entropy loss at the end of epoch {}: {}".format(n, train_err))
                    print("classification error at the end of epoch {}: {}".format(n, class_err))
                    fname = file_name_gen('NN_model_h1_{}_h2_{}_epoch_{}.pkl'.format(nn.hiddenDim[0], nn.hiddenDim[1], n))
                    fp = os.path.join(save_directory, fname)
                    nn.save_model(fp)

        if valid:
            nn.numData = valid[0].shape[0]
            errors = show_error(save_directory, nn, n, [data, target], valid, return_err=return_err)

    print("End of train matrix process.")

    if return_err:
        return errors
    else:
        return None
