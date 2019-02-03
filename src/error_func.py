


def show_error(nn, epoch, train, valid, test):
    '''
    calculattes error in matrix mode
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

    # Test
    nn.numData = test[0].shape[0]

    nn.fprop(test[0], mode='matrix')
    test_loss = nn.errorRate(test[1].T, mode='matrix')
    test_err = classErr(np.argmax(test[1], axis = 1), nn.prediction)

    # Write to log file
    with open('errors.txt', 'a+') as fp:
        line = '{},{},{},{},{},{},{}\n'.format(epoch, training_loss, training_err,
                                             valid_loss, valid_err, test_loss, test_err)
        fp.write(line)
