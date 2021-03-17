import numpy as np


def sigmoid(x):
    # can safely ignore RuntimeWarning: overflow encountered in exp
    return 1 / (1 + np.exp(-x))
