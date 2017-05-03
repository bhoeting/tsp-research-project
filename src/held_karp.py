import operator
from itertools import combinations


def held_karp(distance_matrix):
    """
    Solve the TSP using the held-karp algorihtm.  
    """
    # Distance helper function. This is necessary
    # since we have a diagonal distance matrix.
    def dist(i, j):
        return distance_matrix[min(i, j)][max(i, j)]

    # The cache will map a path S, parent node P, and total 
    # distance d to the optimal subpath of S - {P}, its parent
    # node, and its distance. The last added item will contain
    # the optimal distance, and we can backtrace through the
    # cache to obtain the optimal path.
    cache = {}
    n = len(distance_matrix) - 1

    # Initialize ranges
    subset_sizes = range(1, n - 1)
    destinations = range(2, n + 1)

    # Add the base cases to the cache (i.e. the distance of
    # each destination node to the starting node).
    for d in destinations:
        cache[(d, ())] = (distance_matrix[1][d], (1, ()))

    # Build the cache for each set where the length > 0
    for s in subset_sizes:
        for d in destinations:
            for subset in combinations(destinations, s):
                if d in subset:
                    continue
                candidates = []
                for i, test_subset in enumerate(combinations(subset, s - 1)):
                    k = subset[s - i - 1]
                    distance = cache[(k, test_subset)][0] + dist(d, k)
                    candidates.append((distance, (k, test_subset)))
                    
                cache[(d, subset)] = min(
                    candidates, key=operator.itemgetter(0))
                
    # Compute the final optimal subpath.
    candidates = []
    for i, subset in enumerate(combinations(destinations, n - 2)):
        d = destinations[n - 2 - i]
        candidates.append((cache[(d, subset)][0] + dist(1, d), (d, subset)))

    # Cache the final optimal subpath. Keep track of the key
    # in traveler.
    traveler = (1, tuple(destinations))
    cache[traveler] = min(candidates, key=operator.itemgetter(0))

    # Backtrack through the cache, starting with traveler,
    # to build the optimal subpath.
    path = []
    while traveler is not None:
        path.append(traveler[0])
        if traveler in cache:
            traveler = cache[traveler][1]
        else:
            traveler = None

    return tuple(reversed(path))
