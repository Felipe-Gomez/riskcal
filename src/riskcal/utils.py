import numpy as np


def _ensure_array(x):
    is_scalar = isinstance(x, (int, float))
    if is_scalar:
        return np.asarray([x]), is_scalar
    return np.asarray(x), False
