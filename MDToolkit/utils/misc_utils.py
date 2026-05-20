import math
import re

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
