import numpy as np

from src.utils import mnist_reader


class LoadData:
    def __init__(self):
        self.addOnes = False
        self.data_path = '/data/'

    def convertTarget(self, targetValues):
        # Convert to one-hot encoding
        numClasses = np.max(targetValues) + 1
        return np.eye(numClasses)[targetValues]

    def data_import(self):
        # Split into train/validation/test
        X_train, y_train = mnist_reader.load_mnist('../../data/mnist', kind='train')
        X_test, y_test = mnist_reader.load_mnist('../../data/mnist', kind='t10k')
        X_valid = X_test[0:5000]
        y_valid = y_test[0:5000]
        X_test = X_test[5000:10000]
        y_test = y_test[5000:10000]
        print(y_test.shape)

        # Get one-hot encoding
        y_valid = self.convertTarget(y_valid)
        y_test = self.convertTarget(y_test)
        y_train = self.convertTarget(y_train)

        return X_train, y_train, X_valid, y_valid, X_test, y_test
