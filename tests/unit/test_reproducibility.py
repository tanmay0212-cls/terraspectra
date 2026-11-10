"""
Test reproducibility and deterministic pseudo-random seed configuration.
"""

import random
import numpy as np
from terraspectra.core.utils import set_seed


def test_numpy_reproducibility():
    set_seed(12345)
    seq1 = np.random.normal(size=20)

    set_seed(12345)
    seq2 = np.random.normal(size=20)

    np.testing.assert_array_equal(seq1, seq2)


def test_python_random_reproducibility():
    set_seed(999)
    val1 = [random.random() for _ in range(10)]

    set_seed(999)
    val2 = [random.random() for _ in range(10)]

    assert val1 == val2
