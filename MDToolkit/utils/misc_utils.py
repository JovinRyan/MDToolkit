import math
import re
import decimal
from itertools import islice

def is_real_float(value):

    try:
        x = float(value)
    except (ValueError, TypeError):
        return False

    return math.isfinite(x)

def is_strict_int(value):
    if isinstance(value, float):
        return False  # reject whole-number floats like 0.0, 1.0

    if isinstance(value, str):
        if not re.fullmatch(r"-?\d+", value.strip()):
            return False

    try:
        x = float(value)
    except (ValueError, TypeError):
        return False

    return math.isfinite(x) and x.is_integer()

def check_unique(lst):
    seen = set()
    for item in lst:
        if item in seen:
            return False  # Duplicate found, exit early
        seen.add(item)
    return True

def count_decimals(num):
    # Using decimal module to avoid floating point representation issues
    return abs(decimal.Decimal(str(num)).as_tuple().exponent)


def sort_atom_list_by_index(atom_list : list):
    '''
    '''
    atom_list.sort(key=lambda atom : atom.id)

    return atom_list

def get_n_even_chunks(list_object, n_chunks=10, alternate_trimming = True):
    data = list(list_object)

    n = len(data)
    remainder = n % n_chunks

    left = 0
    right = n - 1

    while remainder != 0:
        if alternate_trimming:
            left += 1
        else:
            right -= 1

        alternate_trimming = not alternate_trimming
        remainder -= 1

    trimmed = data[left:right + 1]

    chunk_size = len(trimmed) // n_chunks

    return [
        trimmed[i * chunk_size:(i + 1) * chunk_size]
        for i in range(n_chunks)
    ]

def rotation_matrix(angles):
    '''
    '''

    x, y, z = np.deg2rad(angles)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx