import numpy as np

def reshape_class_vector(vector):
    result = np.zeros((vector.shape[0], 3))
    for i in range(vector.shape[0]):
        val = vector[i]
        result[i, val] = 1
    return result
