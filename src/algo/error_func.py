import numpy as np
import src.utils.cmd_util as cmd_util


def classErr(target, predicted):
    '''
    not class dependent
    target must NOT be in one hot
    '''
    cnt = 0

    for i in range(target.shape[0]):
        if target[i] != predicted[i]:
            cnt += 1
    return float(cnt) / target.shape[0]


def show_error(save_directory, nn, epoch, train, valid): # test):
    '''
    calculates error in matrix mode
    '''
    # Train
    nn.numData = train[0].shape[0]

    nn.fprop(train[0], mode='matrix')
    training_loss = nn.errorRate(train[1].T, mode='matrix')
    training_err = classErr(np.argmax(train[1], axis = 1), nn.prediction)


    # Valid
    nn.numData = valid[0].shape[0]
    nn.fprop(valid[0], mode='matrix')

    valid_loss = nn.errorRate(valid[1].T, mode='matrix')
    valid_err = classErr(np.argmax(valid[1], axis  =1 ), nn.prediction)
    print("valid_err", valid_err)

    # Test
    # nn.numData = test[0].shape[0]
    #
    # nn.fprop(test[0], mode='matrix')
    # test_loss = nn.errorRate(test[1].T, mode='matrix')
    # test_err = classErr(np.argmax(test[1], axis = 1), nn.prediction)



    if valid: # or test:
        print("save train valid test")
        cmd_util.save_errors(save_directory, epoch, training_loss, training_err,
                         valid_loss, valid_err) # test_loss, test_err
    #
    # else:
    #     # Write only the training loss to log file
    #     # old
    #     with open(save_directory, 'a+') as fp:
    #         line = '{},{},{}\n'.format(epoch, training_loss, training_err)
    #         fp.write(line)