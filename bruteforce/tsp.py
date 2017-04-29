import sys
import math
import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt


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


def find_distance(point1, point2):
    return math.sqrt(
            ((point2[1] - point1[1]) ** 2) +
            ((point2[0] - point1[0]) ** 2))


def path_distance(path, points):
    distance = find_distance(points[path[-1]], points[path[0]])
    for index in range(len(path)):
        if index == len(path) - 1:
            break
        distance += find_distance(points[path[index]], points[path[index+1]])

    return distance


def find_tsp(points):
    # Grab the starting key and remove the
    # starting point from the points dict
    starting_index = next(iter(points.keys()))
    destination_points = points.copy()
    del destination_points[starting_index]

    # Generate all the permutations of the destination points.
    # The starting point will be inserted at the beginnging of
    # the paths before finding the smallest distance
    permutations = itertools.permutations(destination_points)

    # Find the path with the least distance
    shortest_path = None
    smallest_distance = sys.maxsize
    for path in permutations:
        path = (starting_index,) + path
        test_distance = path_distance(path, points)
        if test_distance < smallest_distance:
            smallest_distance = test_distance
            shortest_path = path

    # Create a graph and add each point
    G = nx.Graph()
    G.add_node(starting_index, pos=points[starting_index])
    for label in shortest_path:
        G.add_node(label, pos=points[label])

    # Add the edges
    G.add_edge(shortest_path[-1], starting_index)
    for index in range(len(shortest_path)):
        if index == len(shortest_path) - 1:
            break
        else:
            G.add_edge(shortest_path[index], shortest_path[index+1])

    return G


def main():
    # Read the TSP data
    all_points = read_tsp('test.tsp')

    # Grab 6 random points
    points = {}
    indicies = random.sample(range(1, len(all_points)), 6)
    for i in indicies:
        points[i] = all_points[i]

    # Generate TSP graph
    graph = find_tsp(points)

    # Draw it
    nx.draw(graph,
            nx.get_node_attributes(graph, 'pos'),
            with_labels=False,
            node_size=20)

    plt.show()


if __name__ == "__main__":
    main()
