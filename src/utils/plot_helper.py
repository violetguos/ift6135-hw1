import matplotlib.pyplot as plt
import numpy as np




def plot_learning_curves(log_file):
    with open(log_file, 'r') as fp:
        info = fp.readlines()

    epochs = np.arange(len(info))

    training_loss, valid_loss = [], []
    training_err, valid_err = [], []

    for line in info:
        split_line = line.split(',')
        training_loss.append(float(split_line[1]))
        training_err.append(float(split_line[2]))
        valid_loss.append(float(split_line[3]))
        valid_err.append(float(split_line[4]))

    # Plot
    plt.title("average cross entropy loss")
    plt.plot(epochs, training_loss, c='blue', linestyle='solid', label='train loss')
    plt.plot(epochs, valid_loss, c='green', linestyle='solid', label='valid loss')
    plt.xlabel('number of epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')

    plt.savefig("average cross entropy loss" + ".png")

    plt.show()
    plt.close()
    plt.clf()

    plt.title("classification errors")
    plt.plot(epochs, training_err, c='blue', linestyle='dashed', label='train error')
    plt.plot(epochs, valid_err, c='green', linestyle='dashed', label='valid error')
    plt.xlabel('number of epoch')
    plt.ylabel('error')
    plt.legend(loc='best')

    plt.savefig("average cross entropy error" + ".png")


    plt.show()
    plt.close()
    plt.clf()