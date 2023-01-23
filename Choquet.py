import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB, quicksum

from utils import *


# -------- Choquet LP -------- #

def choquet_lp(n, p, costs, utilities, mobius_masses, combinations=None):

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

    if combinations == None:  
        # Avoir toutes les combinaisons de projets possibles
        objectives_list = [i for i in range(n)]
        combinations = powerset(objectives_list)

    try:
        # Create a new model
        m = gp.Model("Choquet")

        # y: variables that indicate value obtained for each combinations of objectives
        y = m.addMVar(shape=2**n, vtype=GRB.CONTINUOUS, name="y")

        # Set objective
        m.setObjective(mobius_masses @ y, GRB.MAXIMIZE)

        # z: binary variables z to indicate whether a project is selected or not
        z = m.addMVar(shape=p, vtype=GRB.BINARY, name="x")

        # The sum of the costs of the selected projects must be within the budget
        b = sum(costs) / 2  # budget
        m.addConstr(costs @ z <= b, name="budget")

        # The value y_A of a subset of objectives A is the sum of the utilities of the selected projects for those objectives
        for subset_index, subset_obj in enumerate(combinations):
            for i in subset_obj:
                m.addConstr(quicksum(utilities[i-1][j] * z[j] for j in range(p)) >= y[subset_index], name="y_"+str(subset_index)+"_"+str(i))

        m.write("choquet.lp")

        # Optimize model
        m.optimize()

        print("Z: ", z.X)
        print("Y: ", y.X)
        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return z.X, m.Runtime
