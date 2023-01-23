import itertools
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
            costs = read_ints(file)
        costs = np.array(costs)

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

def generate_Choquet_problem(nb_objectives, nb_projects):
    """
    Generates utilities, costs and mobius masses for a problem to be resolved with Choquet.
    """

    utilities = np.random.randint(1, 21, size=(nb_objectives, nb_projects))
    costs = np.random.randint(10, 101, size=nb_projects)
    mobius_masses = belief_function_generator(nb_objectives)

    return utilities, costs, mobius_masses

def OWA_weights_generator(n, alpha=None):

    # Random generation of weights
    if alpha == None:
        alpha = random.randint(1, 10)

    if alpha < 1:
        print("Error: alpha must be greater than or equals to 1.") 

    weights = [((n-i+1)/n)**alpha - ((n-i)/n)**alpha for i in range(n)]
    return np.array(weights)

def belief_function_generator(nb_elements):
    """
    Generates a random list of Mobius masses that correspond to a belief function.
    """

    nb_masses = 2**nb_elements

    # Generate masses for all combinations besides the empty set
    # Dirichlet creates a vector of values that are greater than zero and sum to one
    # Note: this means that we won't have any non-empty subsets that have a mass of 0
    mobius_masses = np.random.dirichlet([1 for j in range(nb_masses-1)])
    # Add mass of zero for the empty set
    mobius_masses = np.insert(mobius_masses, 0, 0)

    return mobius_masses

def powerset(full_set):
    """
    Generates the powerset (list of all subsets) of a given iterable.
    """

    combinations = []
    for i in range(len(full_set) + 1):
        combinations.extend(list(itertools.combinations(full_set, i)))

    return combinations

def lorenz_vector(x):

    sorted_x = sorted(x)

    lorenz = []
    for i in range(len(x)):
        lorenz.append(sum(sorted_x[:i+1]))

    return lorenz