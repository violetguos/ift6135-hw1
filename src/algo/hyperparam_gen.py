import numpy as np


class ParamGenerator():
    """
    Generate parameters like number of hidden units, learning rate, N for finite diff, etc
    """

    def __init__(self, seed):
        # for reproducibility
        self.seed = seed

    def countParam(self, hiddenDim):
        inputDim = 784
        outputDim = 10
        return (inputDim * hiddenDim[0] + hiddenDim[1] * hiddenDim[0]
                + outputDim * hiddenDim[1] + inputDim + hiddenDim[0]
                + hiddenDim[1] + outputDim)

    def hiddenUnit(self):
        """
        return number of hidden units in range (0.5M, 1M)
        """
        constraint = False
        # keep generating until something in between 0.5 to 1 M
        while not constraint:
            h1 = np.random.randint(100, 2000)
            h2 = np.random.randint(100, 2000)
            total_param = self.countParam([h1, h2])
            constraint = (0.5 * 10e5) < self.countParam((h1, h2)) and (10e5) > self.countParam((h1, h2))
        return (h1, h2)

    def learningRate(self):
        """
        sample a learning rate
        """
        logLearningRate = np.random.uniform(-7.5, -4.5)
        learningRate = np.exp(logLearningRate)
        return learningRate

    def finite_diff_epsilon(self):
        """
        :return: the epsilon that we perturb, index into the  [0] to make it an int
        """
        i_arr = np.arange(0, 6)
        print(i_arr)
        k_arr = np.arange(1, 6)
        i = np.random.choice(i_arr, 1)
        k = np.random.choice(k_arr, 1)
        epsilon = k * (10 ** i)
        return epsilon[0]
