import operator
from common import *
from itertools import permutations


def brute_force(distance_matrix):
    """
    Solve the TSP using the brute force method. That
    is, generate each possible path and choose
    the one with the lowest total distance.
    """
    n = len(distance_matrix)
    destinations = range(2, n)
    paths = permutations(destinations)

    candidates = []
    for path in paths:
        path = (1,) + path + (1,)
        candidates.append((
            calc_path_distance(path, distance_matrix),
            path))

    return min(candidates, key=operator.itemgetter(0))[1]
