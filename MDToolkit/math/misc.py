import numpy as np 

def split_arrays_by_overlap(x : np.array, ys: list[np.array]):
    '''
    '''

    overlap_idxs = np.where(np.diff(x) == 0)[0]
    split_idxs = [idx + 1 for idx in overlap_idxs]

    split_x = np.split(x, split_idxs)

    split_ys = [np.split(arr, split_idxs) for arr in ys]

    return split_x, split_ys