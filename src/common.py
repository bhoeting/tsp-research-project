import math
import networkx as nx
from pandas import DataFrame
import matplotlib.pyplot as plt


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
    matrix = [[0] * n for _ in range(n)]

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
        point1 = min(path[i - 1], path[i])
        point2 = max(path[i - 1], path[i])
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
    edge_matrix = [[0] * path_length for _ in range(path_length)]
    for i in range(1, path_length):
        point1 = min(path[i - 1], path[i])
        point2 = max(path[i - 1], path[i])
        edge_matrix[point1][point2] = 1

    return edge_matrix


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
