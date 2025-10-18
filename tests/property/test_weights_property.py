from hypothesis import given, strategies as st
import numpy as np


@given(st.lists(st.floats(min_value=0, max_value=10), min_size=5, max_size=8))
def test_normalized_weights_sum_to_one(vals):
    arr = np.array(vals, dtype=float)
    if arr.sum() == 0:
        arr = arr + 1.0
    w = arr / arr.sum()
    assert abs(w.sum() - 1.0) < 1e-9
