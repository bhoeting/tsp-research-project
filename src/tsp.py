from common import *
from held_karp import *
from brute_force import *


def main():
    size = 14

    P = read_tsp_instance('test.tsp', size)
    D = create_distance_matrix(P)    

    D = D[:size + 1]
    for i in range(len(D)):
        D[i] = D[i][:size + 1]

    if 1:
        hc_path = held_karp(D)
        test_path(hc_path, D, P) 
    else:
        bf_path = brute_force(D)
        test_path(bf_path, D, P) 


if __name__ == "__main__":
    main()
