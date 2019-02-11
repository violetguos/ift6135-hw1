import argparse
import sys

from src.algo.neuralNet import NeuralNet
from src.algo.hyperparam_gen import ParamGenerator
from src.algo.loadData import LoadData
from src.algo.train_matrix import train_matrix


def run_hyperparams(train_data, train_target, valid_data, valid_target,
                    h1=1024, h2=512, learning_rate=2e-3, init_method='glorot'):
    # Instantiate neural network
    nn = NeuralNet(train_data.shape[1], (h1, h2), 10, train_data.shape[0],
                   init_mode=init_method, eta=learning_rate)
    print('Neural network instantiated.')
    print('Total number of parameters: {}'.format(nn.calculParam()))

    # Train and validate neural network
    errors = train_matrix(nn, train_data, train_target, valid=[valid_data, valid_target], return_err=True)

    # Since we are tuning on validation accuracy, just return that
    _, _, _, valid_err = errors[3]
    return valid_err


def run(args):
    '''
    The idea here is to do hyperparameter tuning by random search.
    The hyperparameter generator will return random hyperparameter configs, which we will run
    until a config is found that surpasses 97% validation accuracy.
    '''
    # save_args(args)  # save command line to a file for reference

    # Load data
    dataloader = LoadData()
    train_data, train_target, valid_data, valid_target, test_data, test_target = dataloader.data_import()

    param_generator = ParamGenerator()
    valid_acc = 0.0
    while valid_acc < 0.97:
        hiddens = param_generator.hiddenUnit()
        h1, h2 = hiddens[0], hiddens[1]
        learning_rate = param_generator.learningRate()
        valid_err = run_hyperparams(train_data, train_target, valid_data, valid_target,
                                    h1=h1, h2=h2, learning_rate=learning_rate)
        valid_acc = 1 - valid_err   # To get accuracy out of 100%
    print('Configuration surpassing 97% validation accuracy found!')
    print('valid_acc:', valid_acc)
    print('h1: {}, h2: {}, learning_rate: {}'.format(h1, h2, learning_rate))


def main(argv):
    # Parse input arguments
    parser = argparse.ArgumentParser(description='MLP with numpy MNIST aim for dat 97')
    parser.add_argument('--save_directory', type=str, default='output/q1_dat_97/', help='output directory')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs')
    parser.add_argument("--h1", type=int, default=1024, help='hidden layer 1')
    parser.add_argument('--h2', type=int, default=512, help='hidden layer 2')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help="eta learning rate")
    parser.add_argument('--init_method', type=str, default='glorot', help='normal, zero, glorot')
    args = parser.parse_args(argv)
    run(args)


if __name__ == '__main__':
    # just use a list or json for now instead of config argparser
    main(sys.argv[1:])
