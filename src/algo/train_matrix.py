from src.algo.error_func import show_error, classErr
from src.algo.sampler import BatchSampler
import numpy as np
from src.utils.cmd_util import save_errors, file_name_gen
import os


def train_matrix(nn, data, target, K, num_epoch, args, valid=None, test=None):
    # Get minibatch
    batchSampler = BatchSampler(data, target, K)
    numBatch = data.shape[0] // K

    save_freq = 1#int(num_epoch/5)

    # fixed name for each run for logging errors
    fname_prefix = file_name_gen('NN_model_h1_{}_h2_{}_{}'.format(nn.hiddenDim[0], nn.hiddenDim[1], args.init_method))
    fname_errs = fname_prefix + '.txt'


    print("number of batch in train matrix", numBatch)
    for n in range(num_epoch):
        for i in range(numBatch):
            # Do descent and update params - this is one epoch
            batchData, batchTarget = batchSampler.get_batch()
            nn.numData = K
            nn.gradDescentMat(batchData, batchTarget.T)
            nn.updateParams()
            # print("***************")
            # print(nn.W_3)
            # print(nn.W_2)
            # print(nn.W_1)
            # print()

        if n == (num_epoch - 1) or n % save_freq == 0:
            print(':) saving')
            fname = fname_prefix + '_epoch_{}_rerun.pkl'.format(n+22)
            fp = os.path.join(args.save_directory, fname)
            nn.save_model(fp)
        if valid:
            nn.numData = valid[0].shape[0]

        fp = os.path.join(args.save_directory, fname_errs)

        show_error(fp, nn, n, [data, target], valid)
    print("End of train matrix process.")