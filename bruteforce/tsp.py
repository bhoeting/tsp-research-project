import math
import operator
import networkx as nx
from pandas import DataFrame
import matplotlib.pyplot as plt
from itertools import combinations, permutations


def read_tsp_instance(filename, size):
    """
    Read the data from a standard TSP instance.
    The result is a dictionary that maps an index 
    to a tuple representing the node position.
    """
    points = {}
    reading_points = False

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if 'EOF' in line:
                break
            elif reading_points:
                parts = line.split(' ')
                if int(parts[0]) > size:
                    break
                points[int(parts[0])] = ((int(parts[1]), int(parts[2])))
            elif "NODE_COORD_SECTION" in line:
                reading_points = True
    return points


def create_distance_matrix(points):
    """
    Create a matrix from a dict of points
    where matrix[i, j] is the distance
    between point i and point j, where
    i and j also corrospond to the keys 
    in points
    """
    n = len(points) + 1

    matrix = [[0]*n for _ in range(n)]

    for i, point1 in points.items():
        for j, point2 in points.items():
            if i < j:
                matrix[i][j] = calc_distance(points[i], points[j])

    return matrix


def print_matrix(matrix):
    """
    Print a matrix using the DataFrame
    class from the pandas library.
    """
    print(DataFrame(matrix))


def calc_distance(point1, point2):
    """
    Calculate the distance between two points.
    """
    return math.sqrt(
            ((point2[1] - point1[1]) ** 2) +
            ((point2[0] - point1[0]) ** 2))


def calc_path_distance(path, matrix):
    """
    Calculate the total distance of a path.
    """
    distance = 0
    for i in range(1, len(path)):
        point1 = min(path[i-1], path[i])
        point2 = max(path[i-1], path[i])
        distance += matrix[point1][point2]

    return distance


def create_edge_matrix(path):
    """
    Create edge matrix E from a path
    such that E[i, j] is 1 if an edge
    exists between nodes i and j, or
    0 otherwise.
    """
    path_length = len(path)
    edge_matrix = [[0]*path_length for _ in range(path_length)]
    for i in range(1, path_length):
        point1 = min(path[i-1], path[i])
        point2 = max(path[i-1], path[i])
        edge_matrix[point1][point2] = 1

    return edge_matrix


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

    
def held_karp(distance_matrix):
    """
    Solve the TSP using the held_karp algorihtm.  
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
    subset_sizes = range(1, n-1)
    destinations = range(2, n+1)

    # Add the base cases to the cache (i.e. the distance of
    # each destination node to the starting node).
    for d in destinations:
        cache[(d, ())] = (distance_matrix[1][d], (1, ()))

    # Build the cache for each set where the length > 0
    for size in subset_sizes:
        for d in destinations:
            for subset in combinations(destinations, size):
                if d in subset:
                    continue
                candidates = []
                for i, test_subset in enumerate(combinations(subset, size-1)):
                    k = subset[size - i - 1]
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


def create_graph(edge_matrix, points):
    """
    Use the points and edge matrix to build a graph.
    """
    graph = nx.Graph()
    for point, position in points.items():
        graph.add_node(point, pos=points[point])

    for i in range(1, len(edge_matrix)):
        for j in range(1, len(edge_matrix)):
            if edge_matrix[i][j] == 1:
                graph.add_edge(i, j)
            
    return graph


def test_path(path, distance_matrix, points):
    """
    Create an edge matrix from a path, then
    create a graph from the edge matrix and
    the points, then draw the graph.
    """
    edge_matrix = create_edge_matrix(path)
    graph = create_graph(edge_matrix, points)

    nx.draw(graph,
            nx.get_node_attributes(graph, 'pos'),
            with_labels=False,
            node_size=60)

    plt.show()


def main():
    global size
    size = 14

    P = read_tsp_instance('test.tsp', size)
    D = create_distance_matrix(P)    

    D = D[:size+1]
    for i in range(len(D)):
        D[i] = D[i][:size+1]

    if 1:
        hc_path = held_karp(D)
        test_path(hc_path, D, P) 
    else:
        bf_path = brute_force(D)
        test_path(bf_path, D, P) 

    
if __name__ == "__main__":
    main()
