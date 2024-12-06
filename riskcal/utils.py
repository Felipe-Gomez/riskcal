from typing import Tuple
import numpy as np
from scipy.spatial import ConvexHull


def ensure_array(x):
    if isinstance(x, (int, float)):
        return np.array([x])
    return np.array(x)


def symmetrize_trade_off_curves(
    alpha: np.ndarray, beta1: np.ndarray, beta2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetrize the trade-off curve.
    """
    assert len(alpha) > 3

    # Combine alphas and betas into a single array of points
    points = np.column_stack(
        (
            np.concatenate((alpha, alpha)),
            np.concatenate((beta1, beta2)),
        )
    )

    # Calculate the convex hull
    hull = ConvexHull(points)

    # Get the vertices of the convex hull
    hull_vertices = points[hull.vertices]
    idx = np.argsort(hull_vertices[:, 0])
    return hull_vertices[idx, 0], hull_vertices[idx, 1]
