import os
from time import gmtime, strftime


def file_name_gen(str):
    return strftime("%Y_%m_%d_%H_%M_%S_", gmtime()) + str

def save_args(args):
    # Save argparse arguments to a file for reference
    fname = file_name_gen('args.txt')
    os.makedirs(args.save_directory, exist_ok=True)
    with open(os.path.join(args.save_directory, fname), 'w') as f:
        for k, v in vars(args).items():
            f.write("{}={}\n".format(k, v))


def save_errors(save_directory, epoch, training_loss, training_err, valid_loss, valid_err, test_loss, test_err):
    # Save argparse arguments to a file for reference
    os.makedirs(save_directory, exist_ok=True)
    fname = file_name_gen('train.txt')

    with open(os.path.join(save_directory, fname), 'w') as f:
        line = '{},{},{},{},{},{},{}\n'.format(epoch, training_loss, training_err,
                                             valid_loss, valid_err, test_loss, test_err)
        f.write(line)