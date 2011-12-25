#!/usr/bin/env python

import time
import random
from os import path

def normalize_filename(filename):
    return path.join(path.dirname(__file__), filename)

def print_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
        return res
    return wrapper

def array_to_string(a):
    return [str(x) for x in a]

def normalize(a):
    """Normalize the 1d array a.  Must have non-zero sum"""
    return a / sum(a)


def random_from_dist(ps):
    """Given a list of probabilities (must add to 1), will return an
    index into the list, according to those probabilities.
    Inefficient: does a linear search every time
    """
    r = random.random()
    s = 0
    for i in range(len(ps)):
        s += ps[i]
        if r <= s:
            return i
    raise Exception("random_from_dist shouldn't run off the end of the array")

def custom_flatten(xs):
    """flatten a list that looks like [a,[b,[c,[d,[e]]]]]
    needed because the list can be hundreds of thousands of elements long,
    and the recursion in regular flatten can't handle it."""
    result = []
    while len(xs) != 1:
        result.append(xs[0])
        xs = xs[1]
    if len(xs) == 1:
        result.append(xs[0])
    return result

def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

