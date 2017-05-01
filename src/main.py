from time import time
from common import *
from held_karp import *
from brute_force import *
from termcolor import colored
from prettytable import PrettyTable


def main():
    bf_limit = 11
    sizes = range(4, 19)

    t = PrettyTable(['Cities', 'Brute Force (s)', 'Held-Karp (s)', 'BF/HK'])
    bf_elapsed_prev = 0

    for size in sizes:
        P = read_tsp_instance('test.tsp', size)
        D = create_distance_matrix(P)    

        bf_elapsed = 0
        if size < bf_limit:
            start_time = time()
            brute_force(D)
            bf_elapsed = (time() - start_time)
            bf_elapsed_str = format(bf_elapsed, '.5f')
        else:
            bf_elapsed = bf_elapsed_prev * size
            bf_elapsed_str = colored('{0:,.5f}'.format(bf_elapsed), 'yellow')

        bf_elapsed_prev = bf_elapsed
 
        start_time = time()
        held_karp(D)
        hc_elapsed = (time() - start_time)
        
        t.add_row([
            size,
            bf_elapsed_str,
            '{0:,.5f}'.format(hc_elapsed),
            '{0:,.5f}'.format(bf_elapsed / hc_elapsed)])

    print(t)

if __name__ == "__main__":
    main()
