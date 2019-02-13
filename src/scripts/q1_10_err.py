from src.utils import mnist_reader
import numpy as np
# Print full array for now
#np.set_printoptions(threshold=np.inf)

from src.algo.neuralNet import NeuralNet
from src.algo.train_matrix import train_matrix
import argparse
from src.utils.cmd_util import save_args
from src.algo.loadData import loadData
import sys
import copy
from src.utils.cmd_util import *
import glob
import re
import matplotlib.pyplot as plt
import src.algo.error_func as error_func


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






def run(args):
    save_args(args)  # save command line to a file for reference
    train_data, train_target, valid_data, valid_target, test_data, test_target = data_import()

    train_data = train_data[0:500]
    train_target = train_target[0:500]
    nn = NeuralNet(train_data.shape[1], (args.h1, args.h2), 10, train_data.shape[0],
                          init_mode=args.init_method)

    train_matrix(nn, train_data, train_target, args.batch_size, args.epochs, args)




def plot_graph(epochs, training_loss_glorot, training_loss_zero, training_loss_normal, training_err_glorot,
               training_err_zero, training_err_normal):

    fname = 'Q1_2 different init methods'
    plt.title(fname)

    # Plot
    plt.title("average cross entropy loss")
    plt.plot(epochs, training_loss_glorot, c='blue', linestyle='solid', label='glorot')
    plt.plot(epochs, training_loss_zero, c='green', linestyle='solid', label='zero')
    plt.plot(epochs, training_loss_normal, c='orange', linestyle='solid', label='normal')
    plt.xlabel('number of epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')

    plt.savefig(fname + "loss" + ".png")
    plt.show()
    plt.close()
    plt.clf()

    plt.title("classification errors")
    plt.plot(epochs, training_err_glorot, c='blue', linestyle='dashed', label='glorot')
    plt.plot(epochs, training_err_zero, c='green', linestyle='dashed', label='zero')
    plt.plot(epochs, training_err_normal, c='orange', linestyle='dashed', label='normal')
    plt.xlabel('number of epoch')
    plt.ylabel('error')
    plt.legend(loc='best')

    plt.savefig(fname + "error" + ".png")
    plt.show()
    plt.close()
    plt.clf()


def plot_learning_curves(log_file):
    with open(log_file, 'r') as fp:
        info = fp.readlines()

    epochs = np.arange(len(info))

    training_loss = []
    training_err= []

    for line in info:
        split_line = line.split(',')
        training_loss.append(float(split_line[1]))
        training_err.append(float(split_line[2]))

    return epochs, training_loss, training_err


def main(argv):
    parser = argparse.ArgumentParser(description='MLP with numpy')

    # Output directory
    parser.add_argument('--save_directory', type=str, default='src/scripts/output/q1_3_init_10_err/', help='output directory')


    # Configuration
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs')

    parser.add_argument("--h1", type=int, default=50, help='hidden layer 1')
    parser.add_argument('--h2', type=int, default=50, help='hidden layer 2')

    parser.add_argument('--learning_rate', type=float, default=2e-3, help="eta learning rate")
    parser.add_argument('--init_method', type=str, default='glorot', help='normal, zero, glorot')
    args = parser.parse_args(argv)


    # uncomment the next line to re-train
    # run(args)

    epochs, training_loss_glorot, training_err_glorot = plot_learning_curves(
       'src/scripts/output/q1_3_init_10_err/2019_02_11_18_28_17_NN_model_h1_50_h2_50_glorot.txt')
    epochs, training_loss_zero, training_err_zero = plot_learning_curves(
        'src/scripts/output/q1_3_init_10_err/2019_02_11_18_28_19_NN_model_h1_50_h2_50_zero.txt')
    epochs, training_loss_normal, training_err_normal = plot_learning_curves(
        'src/scripts/output/q1_3_init_10_err/2019_02_11_18_28_18_NN_model_h1_50_h2_50_normal.txt')

    plot_graph(epochs, training_loss_glorot, training_loss_zero, training_loss_normal, training_err_glorot,
               training_err_zero, training_err_normal)


if __name__ == '__main__':
    # uncomment to re-train
    # main(['--init_method=glorot'])
    # main(['--init_method=normal'])
    # main(['--init_method=zero'])

    # plot the graphs from txt files

    epochs, training_loss_glorot, training_err_glorot = plot_learning_curves('output/q1_3_init_10_err/2019_02_11_18_28_17_NN_model_h1_50_h2_50_glorot.txt')
    epochs, training_loss_zero, training_err_zero = plot_learning_curves('output/q1_3_init_10_err/2019_02_11_18_28_19_NN_model_h1_50_h2_50_zero.txt')
    epochs, training_loss_normal, training_err_normal = plot_learning_curves('output/q1_3_init_10_err/2019_02_11_18_28_18_NN_model_h1_50_h2_50_normal.txt')

    plot_graph(epochs, training_loss_glorot, training_loss_zero, training_loss_normal, training_err_glorot,
               training_err_zero, training_err_normal)