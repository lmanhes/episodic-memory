import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pytest

from memory.tree import TreeMemory


@pytest.mark.parametrize("s_m1, a_m1, s, gt",
                         [(12, "right", 8, [2, 1]), (2000, "left", 6, [2, 1]), (20, "right", 20, [1, 1])])
def test_updating(s_m1, a_m1, s, gt):
    tree_memory = TreeMemory(path='test')
    assert tree_memory.graph == None

    tree_memory.update(s_m1, a_m1, s)

    assert len(tree_memory.graph.nodes()) == gt[0]
    assert len(tree_memory.graph.edges()) == gt[1]

    assert tree_memory.graph.has_node(s_m1)
    assert tree_memory.graph.has_node(s)
    assert tree_memory.graph.has_edge(s_m1, s)

    link_attributs = tree_memory.graph.get_edge_data(s_m1, s)[a_m1]
    assert link_attributs['action'] == a_m1
    assert link_attributs['oldness'] == 1
    assert link_attributs['stability'] == 50
    assert link_attributs['weight'] == tree_memory._decay(oldness=1, stability=50)


def test_saving_loading():
    tree_memory = TreeMemory(path='test')

    assert tree_memory.graph == None

    s_m1 = 12
    a_m1 = 'right'
    s = 875
    tree_memory.update(s_m1, a_m1, s)

    tree_memory.save()
    assert os.path.exists('test.graph') and os.path.exists('test.params')

    new_tree_memory = TreeMemory.load('test')
    assert new_tree_memory.graph.has_node(s_m1)
    assert new_tree_memory.graph.has_node(s)
    assert new_tree_memory.graph.has_edge(s_m1, s)

    link_attributs = new_tree_memory.graph.get_edge_data(s_m1, s)['right']
    assert link_attributs['action'] == 'right'
    assert link_attributs['oldness'] == 1
    assert link_attributs['stability'] == 50
    assert link_attributs['weight'] == tree_memory._decay(oldness=1, stability=50)

    os.remove('test.graph')
    os.remove('test.params')


def test_memory_decay():
    tree_memory = TreeMemory(path='test')

    # first update
    tree_memory.update(12, 'right', 13)
    link_attributs = tree_memory.graph.get_edge_data(12, 13)['right']
    oldness_1 = link_attributs['oldness']
    stability_1 = link_attributs['stability']
    weight_1 = link_attributs['weight']
    assert oldness_1 == 1
    assert stability_1 == 50
    assert weight_1 == tree_memory._decay(oldness=1, stability=50)

    # second update of same sequence
    tree_memory.update(12, 'right', 13)
    link_attributs = tree_memory.graph.get_edge_data(12, 13)['right']
    oldness_2 = link_attributs['oldness']
    stability_2 = link_attributs['stability']
    weight_2 = link_attributs['weight']
    assert oldness_2 == 1
    assert stability_2 == 51
    assert weight_2 == tree_memory._decay(oldness=1, stability=51)

    # third update of new sequence
    tree_memory.update(12, 'right', 14)
    link_attributs = tree_memory.graph.get_edge_data(12, 14)['right']
    oldness_3 = link_attributs['oldness']
    stability_3 = link_attributs['stability']
    weight_3 = link_attributs['weight']
    assert oldness_3 == 1
    assert stability_3 == 50
    assert weight_3 == tree_memory._decay(oldness=1, stability=50)

    # results after 3 steps
    link_attributs = tree_memory.graph.get_edge_data(12, 13)['right']
    assert link_attributs['oldness'] == 2
    assert link_attributs['stability'] == 51
    assert link_attributs['weight'] == tree_memory._decay(oldness=2, stability=51)


