import sys
import math
import operator
from itertools import combinations, permutations, chain 
import networkx as nx
import matplotlib.pyplot as plt
from pandas import DataFrame


def read_tsp(filename):
    points = {}
    reading_points = False

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if 'EOF' in line:
                break
            elif reading_points:
                parts = line.split(' ')
                points[int(parts[0])] = ((int(parts[1]), int(parts[2])))
            elif "NODE_COORD_SECTION" in line:
                reading_points = True
    return points


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


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
                matrix[i][j] = find_distance(points[i], points[j])

    return matrix


def print_matrix(matrix):
    print(DataFrame(matrix))


def find_distance(point1, point2):
    return math.sqrt(
            ((point2[1] - point1[1]) ** 2) +
            ((point2[0] - point1[0]) ** 2))


def find_path_distance(path, matrix):
    distance = 0
    for i in range(1, len(path)):
        point1 = min(path[i-1], path[i])
        point2 = max(path[i-1], path[i])
        distance += matrix[point1][point2]

    return distance


def create_edge_matrix(path):
    path_length = len(path)
    edge_matrix = [[0]*path_length for _ in range(path_length)]
    for i in range(1, path_length):
        point1 = min(path[i-1], path[i])
        point2 = max(path[i-1], path[i])
        edge_matrix[point1][point2] = 1

    return edge_matrix


def brute_force(distance_matrix):

    # Use the first point as the starting point
    # (this is not zero because the TPS dataset
    # is indexed at 1.
    start = 1
    n = len(distance_matrix)

    # The destination points will be every point
    # except the starting point.  By "point",
    # we man index.
    destinations = range(start + 1, n)

    # Generate all the permutations of the destination points.
    # It's important to note that the starting index isn't
    # included, as it should be at the start of every path.
    # This will be (n-1)! possible paths.
    paths = permutations(destinations)

    # Find the path with the least distance.
    shortest_path = None
    shortest_distance = sys.maxsize
    for path in paths:
        # Complete the route by add the starting point
        # to the beginning and end of the path.
        path = (start,) + path + (start,)

        # If the path is shorter than all other
        # test paths, set it as the shortest path.
        distance = find_path_distance(path, distance_matrix)
        if distance < shortest_distance:
            shortest_path = path
            shortest_distance = distance

    return shortest_path

    
def held_karp(distance_matrix):
    cache = {}
    n = len(distance_matrix) - 1

    # Distance helper function
    def dist(i, j):
        return distance_matrix[min(i, j)][max(i, j)]

    print_matrix(distance_matrix)

    def print_cache():
        print()
        for key, value in reversed(list(cache.items())):
            print(key, "=>", value)
        print()

    # Initialize ranges
    subset_sizes = range(1, n-1)
    destinations = range(2, n+1)

    # Add the base cases to the cache
    for d in destinations:
        cache[(d, ())] = (distance_matrix[1][d], (1, ()))

    # Build the cache for each set with length > 0
    for size in subset_sizes:
        for d in destinations:
            for subset in combinations(destinations, size):
                if d in subset:
                    continue
                candidates = []
                for index, test_subset in enumerate(
                        combinations(subset, size-1)):
                    k = subset[size - index - 1]
                    distance = cache[(d, test_subset)][0] + dist(d, k)
                    candidates.append((distance, (k, test_subset)))
                    
                cache[(d, subset)] = min(
                        candidates,
                        key=operator.itemgetter(0))
                
    # Add the last solution to the cache
    candidates = []
    for index, subset in enumerate(combinations(destinations, n - 2)):
        k = destinations[n-2-index]
        candidates.append((cache[(k, subset)][0] + dist(1, k), (k, subset)))
    last = cache[(1, tuple(destinations))] = min(
            candidates,
            key=operator.itemgetter(0))
 
    print_cache()

    # Backtrack through the cache and create an
    # the optimal path
    path = [1]
    while last[1] in cache:
        path.append(cache[last[1]][1][0])
        last = cache[last[1]]

    return path


def create_graph(edge_matrix, points):
    # Create a graph and add each point.
    graph = nx.Graph()
    for point, position in points.items():
        graph.add_node(point, pos=points[point])

    # Add the edges
    for i in range(1, len(edge_matrix)):
        for j in range(1, len(edge_matrix)):
            if edge_matrix[i][j] == 1:
                graph.add_edge(i, j)
            
    return graph


def test_path(path, distance_matrix, points):
    edge_matrix = create_edge_matrix(path, distance_matrix)
    graph = create_graph(edge_matrix, points)

    nx.draw(graph,
            nx.get_node_attributes(graph, 'pos'),
            with_labels=True,
            node_size=120)
    plt.show()


def main():
    # Read the TSP data
    P = read_tsp('test.tsp')
    # Create the distance matrix
    D = create_distance_matrix(P)    

    # Reduce the matrix to 8x8 as
    # the BF is too slow for any
    # additional verticies 
    global size
    size = 7
    D = D[:size]
    P = P[:size]
    for i in range(len(D)):
        D[i] = D[i][:size]

    bf_path = brute_force(D)
    test_path(bf_path, D, P) 
    # path2 = held_karp(matrix)

    # print(path1)
    # print(path2)

    return
    
    
if __name__ == "__main__":
    main()
