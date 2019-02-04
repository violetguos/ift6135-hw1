class loadData:
    import numpy as np
    def __init__(self):
        self.addOnes = False
        self.data_path = '/data/'

    def convertTarget(self, targetValues):
        # Convert to one-hot encoding
        numClasses = np.max(targetValues) + 1
        return np.eye(numClasses)[targetValues]

    def loadNumData(self, data, target):
        # Split into train/validation/test
        np.random.seed(6390)
        randIndices = np.random.permutation(data.shape[0])
        data, target = data[randIndices], target[randIndices]

        div1 = int(math.floor(0.8 * data.shape[0]))
        div2 = int(math.floor(0.9 * data.shape[0]))
        trainData, trainTarget = data[:div1], target[:div1]
        validData, validTarget = data[div1:div2], target[div1:div2]
        testData, testTarget = data[div2:], target[div2:]

        # Get one hot encoding
        trainTarget = self.convertTarget(trainTarget)
        validTarget = self.convertTarget(validTarget)
        testTarget = self.convertTarget(testTarget)

        return trainData, trainTarget, validData, validTarget, testData, testTarget
