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
