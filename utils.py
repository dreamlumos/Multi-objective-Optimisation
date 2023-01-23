import numpy as np
import random

# -------- UTILS -------- #

def read_str(file): return file.readline().strip()

def read_int(file): return int(read_str(file))

def read_ints(file): return list(map(int, read_str(file).split()))

def parse_problem(filepath, problem_type='OWA'):
    """
    :param filepath: path to the file containing the problem
    :param problem_type: OWA / Choquet (case-insensitive)
    """

    with open(filepath) as file:
        
        nb_objectives = read_int(file)
        nb_choices = read_int(file)
        
        utilities = []
        for i in range(nb_objectives):
            utilities.append(read_ints(file))
        utilities = np.array(utilities)

        costs = []
        if problem_type.casefold() == 'Choquet'.casefold():
            costs.append(read_ints(file))

    return nb_objectives, nb_choices, utilities, costs

def parse_OWA_problem(filepath):
    nb_agents, nb_items, utilities, _ = parse_problem(filepath, problem_type='OWA')
    return nb_agents, nb_items, utilities

def parse_Choquet_problem(filepath):
    nb_objectives, nb_choices, utilities, costs = parse_problem(filepath, problem_type='Choquet')
    return nb_objectives, nb_choices, utilities, costs

def generate_OWA_problem(nb_agents, nb_items):
    """
    Generates utilities for a problem to be resolved with OWA.
    """

    return np.random.randint(50, size=(nb_agents, nb_items))

def OWA_weights_generator(n, alpha=None):

    # Random generation of weights
    if alpha == None:
        alpha = random.randint(1, 10)

    if alpha < 1:
        print("Error: alpha must be greater than or equals to 1.") 

    weights = [((n-i+1)/n)**alpha - ((n-i)/n)**alpha for i in range(n)]
    return np.array(weights)

def lorenz_vector(x):

    sorted_x = sorted(x)

    lorenz = []
    for i in range(len(x)):
        lorenz.append(sum(sorted_x[:i+1]))

    return lorenz