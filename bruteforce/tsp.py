import sys
import math
import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from pandas import *


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


def find_tsp(distance_matrix):

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
    paths = itertools.permutations(destinations)

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

    # Using the shortest path, we will create
    # another matrix where matrix[i][j] will
    # either be 1 if there is an edge between
    # i and j, or 0 otherwise.
    return create_edge_matrix(shortest_path)
    

def create_graph(edge_matrix, points):
    # Create a graph and add each point.
    graph = nx.Graph()
    for point, position in points.items():
        if point >= 8: break
        graph.add_node(point, pos=points[point])        

    # Add the edges
    for i in range(1, len(edge_matrix)):
        for j in range(1, len(edge_matrix)):
            if edge_matrix[i][j] == 1:
                graph.add_edge(i, j)
            
    return graph


def main():
    # Read the TSP data
    points = read_tsp('test.tsp')

    # Create the matrix
    matrix = create_distance_matrix(points)    

    # Reduce the matrix to 8x8 as
    # the BF is too slow for any
    # additional verticies 
    size = 8
    matrix = matrix[:size]
    for i in range(len(matrix)):
        matrix[i] = matrix[i][:size]
 
    # Create the edge matrix
    edge_matrix = find_tsp(matrix)

    # Create the graph
    graph = create_graph(edge_matrix, points)

    # Draw it
    nx.draw(graph,
            nx.get_node_attributes(graph, 'pos'),
            with_labels=True,
            node_size=120)

    plt.show()


if __name__ == "__main__":
    main()
