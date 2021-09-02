def cm2inch(*tupl):
    """
    Converts reasonable units (cm) into terrible units (inches).
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)