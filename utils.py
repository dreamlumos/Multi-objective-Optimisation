import numpy as np
import random

# -------- UTILS -------- #

def read_str(file): return file.readline().strip()

def read_int(file): return int(read_str(file))

def read_ints(file): return list(map(int, read_str(file).split()))

def parse_OWA_problem(filepath):

    with open(filepath) as file:
        nb_items = read_int(file)
        nb_agents = read_int(file)
        utilities = []

        for i in range(nb_agents):
            utilities.append(read_ints(file))

        utilities = np.array(utilities)

    return nb_agents, nb_items, utilities

def generate_OWA_problem(nb_agents, nb_items):
    """
    Generates utilities for a problem to be resolved with OWA.
    """

    return np.random.randint(30, size=(nb_agents, nb_items))

def OWA_weights_generator(n, alpha=None):

    # Random generation of weights
    if alpha == None:
        alpha = random.randint(1, 10)

    if alpha < 1:
        print("Error: alpha must be greater than or equals to 1.") 

    weights = [((n-i+1)/2)**alpha - ((n-i)/n)**alpha for i in range(n)]
    return np.array(weights)