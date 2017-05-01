import operator
from common import *
from itertools import permutations


def brute_force(distance_matrix):
    """
    Solve the TSP using the brute force method. That
    is, generate each possible path and choose
    the one with the lowest total distance.
    """
    start = 1
    n = len(distance_matrix)

    destinations = range(start + 1, n)
    paths = permutations(destinations)

    candidates = []
    for path in paths:
        path = (start,) + path + (start,)
        candidates.append((calc_path_distance(path, distance_matrix), path))

    solution = min(candidates, key=operator.itemgetter(0))
    return solution[1]

