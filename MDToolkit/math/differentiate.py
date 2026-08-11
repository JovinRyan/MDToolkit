import numpy as np

def forward_finite_difference(x: np.ndarray, y: np.ndarray):
    '''
    '''
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]

    rep_mask = dx == 0
    rep_indices = np.where(rep_mask)[0]

    dy[rep_indices] += dy[rep_indices + 1]
    dx[rep_indices] += dx[rep_indices + 1]

    derivative = dy / dx

    return np.insert(derivative, 0, np.nan)
