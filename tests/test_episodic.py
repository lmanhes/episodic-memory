import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

from memory.episodic import EpisodicMemory


def test_episodic():
    episodic_memory = EpisodicMemory(base_path='tests', max_size=4, index_percentage_threshold=0.01)
    assert len(episodic_memory) == 0

    state_1 = np.array([0, 1, 2])
    action_1 = 'right'
    state_2 = np.array([3, 4, 5])

    episodic_memory.update(state_1, action_1, state_2)

    assert len(episodic_memory) == 2
    assert len(episodic_memory.tree_memory.graph.nodes()) == 2
    assert len(episodic_memory.tree_memory.graph.edges()) == 1

    action_2 = 'left'
    episodic_memory.update(state_1, action_2, state_2)

    assert len(episodic_memory) == 2
    assert len(episodic_memory.tree_memory.graph.nodes()) == 2
    assert len(episodic_memory.tree_memory.graph.edges()) == 2

    state_3 = np.array([3, 4, 5])
    action_3 = 'right'
    state_4 = np.array([0, 1, 2])

    episodic_memory.update(state_3, action_3, state_4)

    assert len(episodic_memory) == 2
    assert len(episodic_memory.tree_memory.graph.nodes()) == 2
    assert len(episodic_memory.tree_memory.graph.edges()) == 3

    state_5 = np.array([0.2, 5.9, 3.3])
    action_5 = 'left'
    state_6 = np.array([0.9, 0.1, 2.4])

    episodic_memory.update(state_5, action_5, state_6)

    assert len(episodic_memory) == 4
    assert len(episodic_memory.tree_memory.graph.nodes()) == 4
    assert len(episodic_memory.tree_memory.graph.edges()) == 4

    episodic_memory.save()

    assert os.path.exists('tests/episodic_memory.params')

    assert os.path.exists('tests/index_memory.params')
    assert os.path.exists('tests/index_memory.index')

    assert os.path.exists('tests/tree_memory.params')
    assert os.path.exists('tests/tree_memory.graph')

    os.remove('tests/episodic_memory.params')
    os.remove('tests/index_memory.params')
    os.remove('tests/index_memory.index')
    os.remove('tests/tree_memory.params')
    os.remove('tests/tree_memory.graph')