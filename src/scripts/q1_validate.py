# src/scripts/output/q1_dat_97/2019_02_12_12_45_53_NN_model_h1_1777_h2_342_epoch_49.pkl
import pickle
from src.utils import mnist_reader
import numpy as np
from src.algo.neuralNet import NeuralNet
from src.algo.train_matrix import train_matrix
import argparse
from src.utils.cmd_util import save_args
import sys
from src.utils.parser_helper import setup_parser, load_json


def convertTarget(targetValues):
    # Convert to one-hot encoding
    numClasses = np.max(targetValues) + 1
    return np.eye(numClasses)[targetValues]

def data_import():
    X_train, y_train = mnist_reader.load_mnist('../../data/mnist', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../../data/mnist', kind='t10k')
    # X_valid = X_test[0:5000]
    # y_valid = y_test[0:5000]
    # X_test = X_test[5000:10000]
    # y_test = y_test[5000:10000]
    print(y_test.shape)

    # convert the targets to one hot
    #y_valid = convertTarget(y_valid)

    y_test = convertTarget(y_test)
    y_train = convertTarget(y_train)

    return X_train, y_train,  X_test, y_test # X_valid, y_valid,  X_test, y_test


def run(args):
    save_args(args)  # save command line to a file for reference
    # train_data, train_target, valid_data, valid_target, test_data, test_target = data_import()

    train_data, train_target, valid_data, valid_target = data_import()

    with open('output/q1_dat_97/2019_02_14_09_15_43_NN_model_h1_494_h2_503_glorot_epoch_21.pkl', 'rb') as jar:
        model = pickle.load(jar)

    nn = model
    print("total number of param in plot_test", nn.calculParam())

    train_matrix(nn, train_data, train_target, args.batch_size, args.epochs, args, [valid_data, valid_target])

def main(argv):

    # uncomment the following for training


    # hidden units, and learning rates are generated randomly, and read from the json file

    parser = argparse.ArgumentParser(description='MLP with numpy')
    parser.add_argument("--h1", type=int, default=494, help='hidden layer 1')
    parser.add_argument("--h2", type=int, default=503, help='hidden layer 1')
    parser.add_argument('--learning_rate', type=float, default=0.0026968445409267646, help="eta learning rate")

    # fixed Configuration
    parser.add_argument('--save_directory', type=str, default='output/q1_dat_97/', help='output directory')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=35, metavar='N', help='number of epochs')
    parser.add_argument('--init_method', type=str, default='glorot', help='normal, zero, glorot')


    args = parser.parse_args(argv)
    run(args)



if __name__ == '__main__':

    main(sys.argv[1:])
