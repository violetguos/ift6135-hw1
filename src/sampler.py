class BatchSampler(object):
    '''
    randomly sample batches without replacement.
    '''
    import numpy as np
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.indices = np.arange(self.num_points)

    def get_batch(self, K = None):
        '''
        Get a random batch without replacement
        '''

        if not K:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.arange(K)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch
