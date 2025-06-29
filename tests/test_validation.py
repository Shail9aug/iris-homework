# tests/test_validation.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from iris import validate_data

import numpy as np

def test_valid_data():
    valid_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    assert validate_data(valid_input) is True

def test_invalid_data():
    assert validate_data([1, 2, 3]) is False  # plain list, invalid
    assert validate_data(np.array([1, 2, 3])) is False  # 1D array, invalid shape
    assert validate_data(np.array([[1, 2]])) is False  # wrong # features
    assert validate_data(np.array([["a", "b", "c", "d"]])) is False  # wrong dtype
