from src.utils import mnist_reader
import numpy as np
from src.algo.neuralNet import NeuralNet
from src.algo.train_matrix import train_matrix
import argparse
from src.utils.cmd_util import save_args
import sys
from src.utils.parser_helper import setup_parser, load_json
from src.utils.plot_helper import plot_learning_curves

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

    return X_train[0:10], y_train[0:10],  X_test[0:10], y_test[0:10] # X_valid, y_valid,  X_test, y_test


def run(args):
    save_args(args)  # save command line to a file for reference
    # train_data, train_target, valid_data, valid_target, test_data, test_target = data_import()

    train_data, train_target, valid_data, valid_target = data_import()


    nn = NeuralNet(train_data.shape[1], (args.h1, args.h2), 10, train_data.shape[0],
                          init_mode=args.init_method, eta=args.learning_rate)
    print("total number of param in plot_test", nn.calculParam())

    #train_matrix(nn, train_data, train_target, args.batch_size, args.epochs, args, [valid_data, valid_target])



def main(argv):

    # uncomment the following for training


    # config = load_json("args2.json")
    #
    # for hyper_parameter_set in config["hyperparam"]:
    #     # hidden units, and learning rates are generated randomly, and read from the json file
    #     parser = setup_parser(hyper_parameter_set, "aim for dat 97")
    #
    #     # fixed Configuration
    #     parser.add_argument('--save_directory', type=str, default='output/q1_dat_97/', help='output directory')
    #     parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    #     parser.add_argument('--epochs', type=int, default=35, metavar='N', help='number of epochs')
    #     parser.add_argument('--init_method', type=str, default='glorot', help='normal, zero, glorot')
    #
    #     args = parser.parse_args(argv)
    #     run(args)
    
    plot_learning_curves('src/scripts/output/q1_dat_97/2019_02_12_23_09_17_NN_model_h1_1461_h2_398_glorot_97_dat1.txt')




if __name__ == '__main__':
    # just use a list or json for now instead of config argparser
    # main(sys.argv[1:])
    plot_learning_curves('src/scripts/output/q1_dat_97/2019_02_12_23_09_17_NN_model_h1_1461_h2_398_glorot_97_dat1.txt')
