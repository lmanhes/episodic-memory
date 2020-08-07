import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pytest
import numpy as np

from memory.utils import create_index


def test_index_creation():
    vectors = np.array([[0,1,2], [3,4,5], [6,7,8]])
    labels = [1, 123, 12]
    search_index = create_index(vectors, labels, space='l2')
    assert np.equal(search_index.get_items([1, 123, 12]), vectors).all()