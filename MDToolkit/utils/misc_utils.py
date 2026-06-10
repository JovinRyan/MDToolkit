import math
import re
import decimal

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

def file_path_to_elements_and_stoichiometries(file_path : str):
    formula = file_path.split("/")[-1].split(".")[0]
    char_list = list(formula)
    element_list = []
    stoich_list = []
    for char in char_list:
        if char.isupper():
            element_list.append(char)
            stoich_list.append(1.0)
        elif char.islower():
            element_list[-1] += char
        elif char.isdigit():
            stoich_list.append(float(char))
            del stoich_list[-2]

    return element_list, stoich_list

def sort_atom_list_by_index(atom_list : list):
    '''
    '''
    atom_list.sort(key=lambda atom : atom.id)

    return atom_list
