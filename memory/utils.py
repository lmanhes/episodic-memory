import hnswlib
import numpy as np
from scipy.spatial.distance import cosine


def create_index(vectors, labels, space='cosine'):
    """

    :param vectors: list of vectors to store (numpy arrays)
    :param labels: list of associated labels (int)
    :param space: 'l2' or 'cosine'
    :return: hnswlib index
    """

    index = hnswlib.Index(space=space, dim=vectors.shape[-1])  # possible options are l2, cosine or ip
    index.init_index(max_elements=vectors.shape[0])
    index.add_items(vectors, labels)
    return index


def save_index(index, path):
    """

    :param index: hnswlib index
    :param path: location to save the index
    :return:
    """

    index.save_index(path)
    del index


def load_index(path, dim, space='cosine', **kwargs):
    """

    :param path: location to load the index
    :param dim: dim of each vector that is stored
    :param space: 'l2' or 'cosine'
    :return: hnswlib index
    """

    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(path, max_elements=kwargs['size'])
    return index


def update_index(index, vectors, labels):
    """

    :param index: hnswlib index
    :param vectors: list of vectors to store (numpy arrays)
    :param labels: list of associated labels (int)
    :return: hnswlib index
    """

    previous_max_element = len(index.get_ids_list())
    index.resize_index(new_size=previous_max_element + vectors.shape[0])
    index.add_items(vectors, labels)
    return index


def distance(a, b, metric="l2"):
    """
    Compute distance (euclidean or cosine) between two vectors

    :param a: numpy 1D array
    :param b: numpy 1D array
    :param metric: either 'cosine' or 'l2'
    :return: distance (float)
    """
    if metric == "l2":
        return np.linalg.norm(a - b)
    else:
        return cosine(a, b)


def perturbate_random(vector, percentage):
    """
    Create a new vector based on the one provided with some of it randomized

    :param vector: numpy 1D array
    :param percentage: [0, 1] percentage of the vector to randomize
    :return: numpy 1D array
    """
    # create a random mask with 'percentage' % of True values
    mask = np.full(vector.shape[-1], False)
    mask[:int(percentage * vector.shape[-1])] = True
    np.random.shuffle(mask)
    mask = mask.astype(bool)

    # replace each True indices of the mask by a random value
    new_vector = np.copy(vector)
    perturbation = np.random.rand(*new_vector.shape)
    new_vector[mask] = perturbation[mask]

    return new_vector


def find_optimal_threshold(dim, perturbation, metric, trials=3000):
    """
    Convert a perturbation value to a threshold value to consider two vectors similar in a
    particular space.

    E.g : I want to know what is the threshold to use to consider two vectors similar if they have
    perturbation % of dissimilarity for the 'l2' space.

    :param dim: size of the vector
    :param perturbation: [0, 1], percentage of dissimilarity to consider two vectors similar
    :param metric: either 'cosine' or 'l2'
    :param trials: number of experiment to calculate the average threshold
    :return: threshold (float)
    """
    results = []
    for _ in range(trials):
        vector = np.random.rand(dim)
        randomized_vector = perturbate_random(vector, perturbation)
        dist = distance(vector, randomized_vector, metric)
        results.append(dist)

    return np.mean(results)