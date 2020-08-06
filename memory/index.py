import pickle
import numpy as np

from .utils import create_index, load_index, update_index, save_index, find_optimal_threshold


class IndexMemory(object):
    """
    Store state vectors and search top-k neighboors
    """

    def __init__(self,
                 path,
                 space='l2',
                 dim=None,
                 size=None,
                 percentage_threshold=0.05,
                 **kwargs):
        """

        :param path: location of the index memory
        :param space: 'l2' or 'cosine'
        :param dim: dimension of state vectors (default: None)
        :param size: number of elements in the index (default: None)
        :param percentage_threshold: similarity threshold for 1-nn as a percentage
        """
        self.space = space
        self.path = path
        self.dim = dim
        self.size = size

        self.percentage_threshold = percentage_threshold
        self.sim_threshold = None

        self.index = None

    def __len__(self):
        return self.size

    def init(self, vectors, labels):
        self.index = create_index(vectors=vectors,
                                  labels=labels,
                                  space=self.space)
        self.dim = vectors.shape[-1]
        self.size = vectors.shape[0]

        self.sim_threshold = find_optimal_threshold(dim=self.dim,
                                                    perturbation=self.percentage_threshold,
                                                    metric=self.space)

    def update(self, vectors, labels):
        """
        Add new state vector(s) in the index

        :param vectors: 1D or 2D numpy array
        :param labels: int or list of int
        :return:
        """
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, 0)

        if self.index is None:
            self.init(vectors=vectors,
                      labels=labels)
        else:
            assert vectors.shape[-1] == self.dim
            update_index(index=self.index,
                         vectors=vectors,
                         labels=labels)
            self.size += vectors.shape[0]

    def forget(self, labels):
        """
        Remove labels to forget from index

        :param labels: list of labels
        :return:
        """
        for l in labels:
            self.index.mark_deleted(l)

        self.size -= len(labels)

    def query(self, vector, k):
        """
        Retrieve k most similar neighboors of the query vector

        :param vector: 1D numpy array
        :param k: number of vector to retrieve
        :return: list of k labels, list of k distances
        """
        # prevent k-query for index size < k
        k = min(k, self.size)

        idxs, dists = self.index.knn_query(data=vector, k=k)

        return idxs[0], dists[0]

    def has_neighbor(self, vector):
        """
        Find if the query vector has a close neighbor in the memory
        'sim_threshold' args defines this behavior

        :param vector: 1D numpy array
        :return: neighbor if exists else None
        """
        idxs, dists = self.index.knn_query(data=vector, k=1)
        if dists[0][0] <= self.sim_threshold:
            return idxs[0][0]

    @classmethod
    def load(cls, path):
        memory_params = pickle.load(open(f'{path}.params', "rb"))
        index_memory = cls(path=path, **memory_params)
        index_memory.index = load_index(path=f'{path}.index',
                                   **memory_params)
        index_memory.sim_threshold = memory_params['sim_threshold']
        return index_memory

    def save(self):
        memory_params = {'dim': self.dim,
                         'space': self.space,
                         'sim_threshold': self.sim_threshold,
                         'percentage_threshold': self.percentage_threshold,
                         'size': self.size}
        pickle.dump(memory_params, open(f'{self.path}.params', "wb" ))
        save_index(index=self.index,
                   path=f'{self.path}.index')
