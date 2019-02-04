import os


def save_args(args):
    # Save argparse arguments to a file for reference
    os.makedirs(args.save_directory, exist_ok=True)
    with open(os.path.join(args.save_directory, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write("{}={}\n".format(k, v))


def save_errors(save_directory, epoch, training_loss, training_err, valid_loss, valid_err, test_loss, test_err):
    # Save argparse arguments to a file for reference
    os.makedirs(save_directory, exist_ok=True)
    with open(os.path.join(args.save_directory, 'train.txt'), 'w') as f:
        line = '{},{},{},{},{},{},{}\n'.format(epoch, training_loss, training_err,
                                             valid_loss, valid_err, test_loss, test_err)
        f.write(line)