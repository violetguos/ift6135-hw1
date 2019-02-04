import numpy as np


def zero_init(d, dh1, dh2, m):
    W_1 = np.zeros((dh1*d))
    W_2 = np.zeros((dh2*dh1))
    W_3 = np.zeros((dh2*m))
    return W_1, W_2, W_3

def glorot_init(d, dh1, dh2, m):
    dl_1 = np.sqrt((6/(d + dh1)))
    W_1 = np.random.uniform((-1)*dl_1, dl_1, (dh1, d))

    dl_2 = np.sqrt(6/(dh1 + dh2))
    W_2 = np.random.uniform((-1)*dl_2, dl_2, (dh2, dh1))

    dl_3 = np.sqrt(6/(dh2 + m))
    W_3 = np.random.uniform((-1)*dl_3, dl_3, (m, dh2))

    return W_1, W_2, W_3

def normal_init(d, dh1, dh2, m):
    W_1 = np.random.normal(0, 1, (dh1, d))
    W_2 = np.random.normal(0, 1, (dh2, dh1))
    W_3 = np.random.normal(0, 1, (m, dh2))
    return W_1, W_2, W_3
