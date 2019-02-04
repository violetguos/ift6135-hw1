from src.utils import mnist_reader
import numpy as np
from src.algo.neuralNet import NeuralNet
from src.algo.train_matrix import train_matrix
from src.algo.hyperparam_gen import ParamGenerator
import argparse
from src.utils.cmd_util import save_args
import sys

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
    plot_test = NeuralNet(train_data.shape[1], (args.h1, args.h2), args.epochs, args.batch_size, args.init_method)
    print("total number of param in plot_test", plot_test.calculParam())
    train_matrix(plot_test, train_data, train_target, args.epochs,args.batch_size,args.save_directory,
                valid=[valid_data, valid_target], test=[test_data, test_target])


def main(argv):
    parser = argparse.ArgumentParser(description='MLP with numpy')

    # Output directory
    parser.add_argument('--save_directory', type=str, default='output/q1', help='output directory')


    # Configuration
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs')

    parser.add_argument("--h1", type =int, default=10, help='hidden layer 1')
    parser.add_argument('--h2', type=int, default=10, help='hidden layer 2')

    parser.add_argument('--learning_rate', type=float, default=2e-3, help="eta learning rate")
    parser.add_argument('--init_method', type=str, default='normal', help='normal, zero, golorot')
    args = parser.parse_args(argv)
    run(args)



if __name__ == '__main__':

    main(sys.argv[1:])

