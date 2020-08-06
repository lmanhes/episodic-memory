import random
import pickle
import os

from .index import IndexMemory
from .tree import TreeMemory


class EpisodicMemory(object):

    def __init__(self, base_path, max_size=1e5, index_percentage_threshold=0.05, vector_dim=3,
                 stability_start=100):
        """
        :param base_path: location of the episodic memory
        :param max_size: max number of different states in the memory
        :param index_percentage_threshold: similarity percentage threshold for 1-nn
        :param vector_dim: size of state vector
        :param stability_start: starting stability parameter for the tree memory
        """
        self.episodic_memory_path = os.path.join(base_path, 'episodic_memory')
        self.max_size = int(max_size)
        self.index_percentage_threshold = index_percentage_threshold
        self.vector_dim = vector_dim
        self.stability_start = stability_start

        self.id_bank = []

        self.index_memory_path = os.path.join(base_path, 'index_memory')
        self.index_memory = IndexMemory(path=self.index_memory_path,
                                        percentage_threshold=index_percentage_threshold,
                                        dim=vector_dim,
                                        space='l2')

        self.tree_memory_path = os.path.join(base_path, 'tree_memory')
        self.tree_memory = TreeMemory(path=self.tree_memory_path, stability_start=stability_start)

        self.forgeted = 0

    def __len__(self):
        return len(self.id_bank)

    def get_states_attributes(self, ids_list=None):
        """
        Get states attributes from memory

        :param ids_list: list of ids to retrieve (optional)
        :return: dict of id: {'attr_1: .., ...}
        """

        nodes = self.tree_memory.get_nodes()

        if ids_list is not None:
            nodes = [n for n in nodes if n[0] in ids_list]

        return {int(n[0]): n[1] for n in nodes}

    def get_raw_states(self, ids_list=None):
        """
        Get raw vector states from memory

        :param ids_list: list of ids to retrieve (optional)
        :return: dict of id: 1D numpy vector (state)
        """

        if ids_list is None:
            ids_list = [n[0] for n in self.tree_memory.get_nodes()]

        vector_list = self.index_memory.index.get_items(ids_list)
        return {int(id): vector for id, vector in zip(ids_list, vector_list)}

    def _assign_id(self):
        """
        Generate a free id between 0 and 'self.max_size'

        :return: id (int)
        """

        if len(self.id_bank) < self.max_size:
            free_ids = list(set(range(self.max_size)) - set(self.id_bank))
            id = random.choice(free_ids)
            self.id_bank.append(id)
            return id

    def update(self, state_m1, action_m1, state):
        """
        Update episodic memory with a new sequence
        Remove the forgeted ones

        :param state_m1: state at t-1 (1D numpy vector)
        :param action_m1: action at t-1 (string)
        :param state: state at t (1D numpy vector)
        :return:
        """

        if len(self.id_bank) == 0:
            state_m1_id = self._assign_id()
            self.index_memory.update(state_m1, state_m1_id)
        else:
            # find if state at t-1 exists and get the associated id
            # if not assign it an id and update the index memory
            state_m1_id = self.index_memory.has_neighbor(state_m1)
            if state_m1_id is None:
                state_m1_id = self._assign_id()
                self.index_memory.update(state_m1, state_m1_id)

        # find if state at t exists and get the associated id
        # if not assign it an id and update the index memory
        state_id = self.index_memory.has_neighbor(state)
        if state_id is None:
            state_id = self._assign_id()
            self.index_memory.update(state, state_id)

        # if there is forgoted nodes remove them from index memory
        forgeted = self.tree_memory.update(state_m1_id, action_m1, state_id)
        self.forgeted += len(forgeted)
        if forgeted: self.index_memory.forget(forgeted)

    @classmethod
    def load(cls, base_path):
        episodic_memory_path = os.path.join(base_path, 'episodic_memory')
        episodic_params = pickle.load(open(f'{episodic_memory_path}.params', "rb"))
        episodic_memory = cls(base_path=base_path, **episodic_params)

        episodic_memory.tree_memory = TreeMemory.load(episodic_memory.tree_memory_path)
        episodic_memory.index_memory = IndexMemory.load(episodic_memory.index_memory_path)

        return episodic_memory

    def save(self):
        episodic_params = {'max_size': self.max_size,
                           'index_percentage_threshold': self.index_percentage_threshold,
                           'vector_dim': self.vector_dim,
                           'stability_start': self.stability_start}
        pickle.dump(episodic_params, open(f'{self.episodic_memory_path}.params', "wb"))
        self.tree_memory.save()
        self.index_memory.save()