import math
import operator
from itertools import combinations, permutations, chain 
import networkx as nx
import matplotlib.pyplot as plt
from pandas import DataFrame


def read_tsp(filename, size):
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
    start = 1
    n = len(distance_matrix)

    print("bf n", n)

    destinations = range(start + 1, n)
    paths = permutations(destinations)

    candidates = []
    for path in paths:
        path = (start,) + path + (start,)
        candidates.append((find_path_distance(path, distance_matrix), path))

    solution = min(candidates, key=operator.itemgetter(0))
    print("bfdistance", solution[0])
    return solution[1]

    
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

    print(list(subset_sizes))
    print(list(destinations))

    # return destinations

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
                for i, test_subset in enumerate(combinations(subset, size-1)):
                    k = subset[size - i - 1]
                    distance = cache[(k, test_subset)][0] + dist(d, k)
                    print("set=", subset, "k=", k, "d=", d)
                    print("distance=", cache[(k, test_subset)][0], "+", dist(d, k))
                    print()
                    candidates.append((distance, (k, test_subset)))
                    
                cache[(d, subset)] = min(
                        candidates, key=operator.itemgetter(0))
                
    # Add the last solution to the cache
    candidates = []
    for i, subset in enumerate(combinations(destinations, n - 2)):
        d = destinations[n - 2 - i]
        candidates.append((cache[(d, subset)][0] + dist(1, d), (d, subset)))

    traveler = (1, tuple(destinations))
    cache[traveler] = min(
            candidates, key=operator.itemgetter(0))
 
    print_cache()

    path = []
    while traveler is not None:
        path.append(traveler[0])
        if traveler in cache:
            traveler = cache[traveler][1]
        else:
            traveler = None


    # Backtrack through the cache and create an
    # the optimal path
    #path = [1]
    #while traveler in cache:
    #    print("appending", cache[traveler[1]][1], "[0]")
    #    path.append(cache[traveler[1]][1][0])
    #    traveler = cache[traveler[1]]
    print()
    print(path)
    print()
    return tuple(reversed(path))


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
    edge_matrix = create_edge_matrix(path)
    graph = create_graph(edge_matrix, points)

    nx.draw(graph,
            nx.get_node_attributes(graph, 'pos'),
            with_labels=True,
            node_size=120)
    plt.show()


def main():
    global size
    size = 10

    P = read_tsp('test.tsp', size)
    D = create_distance_matrix(P)    

    D = D[:size+1]
    for i in range(len(D)):
        D[i] = D[i][:size+1]

    bf_path = brute_force(D)
    print(bf_path)
    
    hc_path = held_karp(D)

    print("harpklarp", hc_path)
    print("brute force", bf_path)

    return

    if 1:
        hc_path = held_karp(D)
        test_path(hc_path, D, P) 
    else:
        bf_path = brute_force(D)
        test_path(bf_path, D, P) 

    
if __name__ == "__main__":
    main()
