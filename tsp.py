import sys
import math
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
    print(point1)
    print(point2)
    return math.sqrt(
            ((point2[1] - point1[1]) ** 2) +
            ((point2[0] - point1[0]) ** 2))


def main():

    # Create the graph
    points_dict = read_tsp('test.tsp')
    G = nx.Graph()

    # Add the nodes
    for index, point in points_dict.items():
        G.add_node(index, pos=(point[0], point[1]))

    # Add edges between each node and its closest node
    # for node_index, point in points_dict.items():

    # closest_distance = sys.maxsize
    # starting_node = 1
    # closest_destination_node = -1
    # for index, point in points_dict.items():
    #     distance = find_distance(points_dict[starting_node], point)
    #     if (distance > 0 and distance < closest_distance):
    #         closest_distance = distance
    #         closest_destination_node = index

    # G.add_edge(starting_node, closest_destination_node)

    nx.draw(
            G,
            nx.get_node_attributes(G, 'pos'),
            with_labels=False,
            node_size=20)

    plt.show()


if __name__ == "__main__":
    main()
