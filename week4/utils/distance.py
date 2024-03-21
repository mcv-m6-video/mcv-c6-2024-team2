import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def l2distance(input1, input2):
    return np.linalg.norm(input1 - input2)


def cossim(input1, input2):
    return cosine_similarity(input1.reshape(1,-1), input2.reshape(1,-1))
