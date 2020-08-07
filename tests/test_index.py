import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

from memory.index import IndexMemory


def test_update():
    index_memory = IndexMemory(path='test', space='l2', percentage_threshold=0.01)
    assert index_memory.index == None

    vectors = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    labels = [1, 123, 1222222]
    index_memory.update(vectors, labels)
    assert index_memory.index != None
    assert set(index_memory.index.get_ids_list()) == {1, 123, 1222222}
    assert index_memory.size == 3

    new_vector = np.array([0, 1, 1])
    new_label = 121
    index_memory.update(new_vector, new_label)
    assert set(index_memory.index.get_ids_list()) == {1, 123, 1222222, 121}
    assert index_memory.size == 4


def test_saving_loading():
    index_memory = IndexMemory(path='test', space='l2', percentage_threshold=0.01)
    vectors = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    labels = [1, 123, 1222222]
    index_memory.update(vectors, labels)

    index_memory.save()
    assert os.path.exists('test.index') and os.path.exists('test.params')

    new_index_memory = IndexMemory.load('test')
    assert new_index_memory.dim == 3
    assert new_index_memory.space == 'l2'
    assert new_index_memory.percentage_threshold == 0.01
    assert new_index_memory.size == 3

    os.remove('test.index')
    os.remove('test.params')


def test_forget():
    index_memory = IndexMemory(path='test', space='l2', percentage_threshold=0.01)

    vectors = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    labels = [1, 123, 1222222]
    index_memory.update(vectors, labels)
    assert len(index_memory) == 3

    index_memory.forget([1222222])
    assert len(index_memory) == 2

    idxs, dists = index_memory.query(np.array([0, 1, 2]), k=3)
    assert len(idxs) == 2

    index_memory.update(np.array([9, 9, 9]), 1222222)
    assert len(index_memory) == 3
    idxs, dists = index_memory.query(np.array([[9, 9, 9]]), k=3)
    assert idxs[0] == 1222222
