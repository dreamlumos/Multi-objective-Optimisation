import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB, quicksum

from utils import *

##############################
###### Does not work yet #####
##############################

# -------- Choquet Graph LP -------- #

def choquet_graph_lp(n, traveling_time, mobius_masses, combinations=None):
    """
    :param n: number of objectives
    :param p: number of projects
    :param costs: costs for each project: [c1, ..., ck] with k in {1, ..., p}
    :param utilities: U
    :param mobius_masses: Mobius masses
    :param combinations: list of combinations of objectives

    :type n: int
    :type p: int
    :type costs: ndarray[int]
    :type utilities: ndarray[int]
    :type mobius_masses: ndarray[float]
    :type combinations: list[tuple[int]]

    :return solution: x
    :rtype: ndarray[int]
    """


    nb_nodes = len(traveling_time[0])

    if combinations == None:  
        # Avoir toutes les combinaisons de projets possibles
        objectives_list = [i for i in range(n)]
        combinations = powerset(objectives_list)

    try:
        # Create a new model
        m = gp.Model("ChoquetGraph")

        # y: variables that indicate value obtained for each combination of scenarios
        y = m.addMVar(shape=2**n, vtype=GRB.CONTINUOUS, name="y")

        # Set objective
        m.setObjective(mobius_masses @ y, GRB.MAXIMIZE)

        # x: binary decision variables to indicate when an arc is taken
        x = m.addMVar(shape=(nb_nodes, nb_nodes), vtype=GRB.BINARY, name="x")

        # Path constraints
        m.addConstrs((x[i,:] == x[:,i] for i in range(nb_nodes)), name="path")
        m.addConstr(x[1,:] * np.ones(nb_nodes) == 1)
        m.addConstr(x[:,nb_nodes-1] * np.ones(nb_nodes) == 1)

        # z: score in each scenario
        z = m.addMVar(shape=n)
        for n in range(n):
            m.addConstrs((z[n] == - x[i] @ traveling_time[n][i] for i in range(nb_nodes)), name="score")

        # The value y_A of a subset of objectives A is the sum of the utilities of the selected projects for those objectives
        for subset_index, subset_obj in enumerate(combinations):
            for i in subset_obj:
                m.addConstr(z[i] >= y[subset_index], name="y_"+str(subset_index)+"_"+str(i))

        m.write("choquet_graph.lp")

        # Optimize model
        m.optimize()

        print("X: ", x.X)
        print("Z: ", z.X)
        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return x.X, m.Runtime

