from src.utils import mnist_reader
import numpy as np
# Print full array for now
#np.set_printoptions(threshold=np.inf)

from src.algo.neuralNet import NeuralNet
from src.algo.train_matrix import train_matrix
from src.algo.hyperparam_gen import ParamGenerator
import argparse
from src.utils.cmd_util import save_args
from src.algo.loadData import loadData
import sys
import copy
from src.utils.cmd_util import *
import glob
import re
import matplotlib.pyplot as plt


def convertTarget(targetValues):
    # Convert to one-hot encoding
    numClasses = np.max(targetValues) + 1
    return np.eye(numClasses)[targetValues]

def data_import():
    X_train, y_train = mnist_reader.load_mnist('../../data/mnist', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../../data/mnist', kind='t10k')
    X_valid = X_test[0:5000]
    y_valid = y_test[0:5000]
    X_test = X_test[5000:10000]
    y_test = y_test[5000:10000]
    print(y_test.shape)

    # convert the targets to one hot
    y_valid = convertTarget(y_valid)
    y_test = convertTarget(y_test)
    y_train = convertTarget(y_train)

    return X_train, y_train, X_valid, y_valid, X_test, y_test




def oneDiff(nn, x_i, y_i, sigma, outer_i):

    # outer_i of 10 elements to perturb
    grad_W2 = nn.grad_W2

    ########### f(x + e)
    nnDebug1 = copy.deepcopy(nn)
    nnDebug1.W_2[0][outer_i] += sigma
    nnDebug1.gradDescentLoop(x_i, y_i, 1)
    err1 = []
    for i in y_i.T:
        err1.append(nnDebug1.errorRate(i, mode='loop'))
    err1 = np.array(err1)

    ########### f(x - e)
    nnDebug2 = copy.deepcopy(nn)
    nnDebug2.W_2[0][outer_i] -= sigma
    nnDebug2.gradDescentLoop(x_i, y_i, 1)
    err2 = []
    for i in y_i.T:
        err2.append(nnDebug2.errorRate(i, mode='loop'))
    err2 = np.array(err2)




    estimate = (err1 - err2) / (sigma*2)
    ratio = estimate / grad_W2[0][outer_i]
    # print("*******************************")
    # print("err1\n", err1)
    # print("err2\n", err2)
    #
    # print("estimate:", estimate)
    # print("grad_W2[0][1]", grad_W2[0][outer_i])
    print('Perturbing an element in W2[0][{}]. Ratio: {}'.format(outer_i, ratio))
    return estimate

def run(args):
    save_args(args)  # save command line to a file for reference
    train_data, train_target, valid_data, valid_target, test_data, test_target = data_import()
    nn = NeuralNet(train_data.shape[1], (args.h1, args.h2), 10, train_data.shape[0],
                          init_mode=args.init_method)
    print("total number of param in plot_test", nn.calculParam())

    x_i = train_data[0:1]
    y_i = train_target[0:1].T

    diff_dict = {}
    for i in range(20):

        nn.gradDescentLoop(x_i, y_i, 1)
        param_gen = ParamGenerator(10) # seed not really used
        N = param_gen.finite_diff_epsilon()
        sigma = 1/N

        estimate_arr = []
        for i in range(0, 10):
            print(type(i))
            r = oneDiff(nn, x_i, y_i, sigma, i)
            estimate_arr.append(r)

        estimate_arr = np.array(estimate_arr).flatten()

        print("estimate_arr")
        print(estimate_arr)
        fname = file_name_gen("_q1_finit_grad_N_" + str(N))
        np.savetxt(args.save_directory +fname+'.txt', estimate_arr)
        grad_10 = nn.grad_W2[0][0:10]
        np.savetxt(args.save_directory +fname+'_real_grad.txt',grad_10)

        # find the maximum gradient diff
        diff = np.abs(estimate_arr-grad_10)

        print("*************")
        print(grad_10)


        diff_max = np.max(diff)

        diff_dict[N] = diff_max

    print("diff_dict")
    print(diff_dict)
    #np.savetxt(args.save_directory +fname+'_diff_grad.txt',grad_10)


def plot_diff(args):
    print(os.getcwd())
    max_diff = []
    n_val_arr = []
    for file in glob.glob(args.save_directory+'newTrial*_diff_grad.txt'):
        # get the numpy array
        grad_diff = np.loadtxt(file)
        print(np.max(grad_diff))
        max_diff.append(np.max(grad_diff))

        # extracts the N value in perturbing, extract as a list of a single str
        n_arr = re.findall("N_(\d+)", file)
        n_str = n_arr[0]
        n_val = float(n_str)
        n_val_arr.append(n_val)

    #return max_diff, n_val_arr
    np.savetxt("new_max_diff.txt", max_diff)
    np.savetxt("new_n_val_arr.txt", n_val_arr)


def plot_final():

    # max_diff = np.loadtxt("new_max_diff.txt")
    # n_val_arr = np.loadtxt("new_n_val_arr.txt")
    #
    # #n_val_arr = 1/ n_val_arr
    # n_val_idx = np.argsort(n_val_arr)
    # print(n_val_idx)
    # max_diff = max_diff[n_val_idx]
    # print(max_diff)

    diff_dict = {20: 0.0, 100000: 0.0, 5: 0.7576155999830192, 40: 0.0, 1: 6.387484313413516, 10000: 0.0, 30000: 0.0, 50: 0.0, 3: 3.536721505027195, 3000: 0.0, 5000: 0.0, 1000: 0.0, 400000: 0.0, 500000: 0.0, 200: 0.0, 300: 0.0}
    lists = sorted(diff_dict.items())  # sorted by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples



    fname = 'Q1_3 N vs gradient diff'
    plt.title(fname)
    plt.xscale('log')
    #plt.yscale('log')
    plt.plot(x, y)
    #plt.legend()


    plt.savefig(fname + ".png")
    plt.show()

    plt.close()
    plt.clf()




def main(argv):
    parser = argparse.ArgumentParser(description='MLP with numpy')

    # Output directory
    parser.add_argument('--save_directory', type=str, default='output/q1_3/', help='output directory')


    # Configuration
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs')

    parser.add_argument("--h1", type=int, default=50, help='hidden layer 1')
    parser.add_argument('--h2', type=int, default=50, help='hidden layer 2')

    parser.add_argument('--learning_rate', type=float, default=2e-3, help="eta learning rate")
    parser.add_argument('--init_method', type=str, default='glorot', help='normal, zero, glorot')
    args = parser.parse_args(argv)

    # uncomment to retrain
    # run(args)
    # plot_diff(args)

    plot_final()


if __name__ == '__main__':

    # uncomment to retrain

    # for i in range(50):
    # main(sys.argv[1:])

    diff_dict = {20: 0.0, 100000: 0.0, 5: 0.7576155999830192, 40: 0.0, 1: 6.387484313413516, 10000: 0.0, 30000: 0.0, 50: 0.0, 3: 3.536721505027195, 3000: 0.0, 5000: 0.0, 1000: 0.0, 400000: 0.0, 500000: 0.0, 200: 0.0, 300: 0.0}
    lists = sorted(diff_dict.items())  # sorted by key, return a list of tuples
    print(lists)
