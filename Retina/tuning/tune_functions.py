## Generator function to return a range of values with a float step
def frange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
