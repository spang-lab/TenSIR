def inside_bounds(values, bounds):
    """
    Returns whether all values lie inside of the bounds.

    :param values: Collection of numbers
    :param bounds: List of size 2 specifying the lower and upper bounds respectively
    :return: Whether all values lie inside the bounds
    """
    lower, upper = bounds
    for v in values:
        if v < lower or v > upper:
            return False
    return True
