import numpy as np

import gurobipy as gp
from gurobipy import GRB, quicksum

from utils import *

# -------- WOWA LP -------- #

def WOWA_LP(n, p, utilities, mobius_masses, one_to_one=True):
    """
    :param n: nb_agents
    :param p: nb_items
    :param utilities: U
    :param weights: [p_1, p_2, ..., p_n] corresponding to importance of each agent
    :param one_to_one: indicates whether only one item is to be attributed per agent

    :type nb_agents: int
    :type nb_items: int
    :type utilities: ndarray[int]
    :type weights: ndarray[int]
    :type one_to_one: bool

    :return solution: x
    :rtype: ndarray[int]
    """

    agents_list = [i for i in range(n)]
    combinations = powerset(agents_list)

    try:
        # Create a new model
        m = gp.Model("WOWA")

        # y: variables that indicate value obtained for each combination of agents
        y = m.addMVar(shape=2**n, vtype=GRB.CONTINUOUS, name="y")

        # Set objective
        m.setObjective(mobius_masses @ y, GRB.MAXIMIZE)

        #### Constraints of the original problem (without linearisation) ####

        # Create binary variables x_ij (if x_ij is 1, the item j is attributed to agent i)
        x = m.addMVar(shape=(n,p), vtype=GRB.BINARY, name="x")

        # For all agents, we sum the value of the items they are attributed
        z = m.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="z")
        m.addConstrs((z[i] == x[i] @ utilities[i] for i in range(n)), name="c_z")

        # We ensure that each item is only attributed once
        nb_attributions_per_item = np.ones(p)
        m.addConstrs((sum(x[:,j]) <= nb_attributions_per_item[j] for j in range(p)), name="c_nbattitems")

        if one_to_one:
            # We ensure that each agent receives only one item
            nb_attributions_per_agent = np.ones(n)
            m.addConstrs((sum(x[i]) <= nb_attributions_per_agent[i] for i in range(n)), name="c_nbattagents")

        #### WOWA and linearisation constraints ####

        # The value y_A of a subset of agents A is the sum of the utilities of the selected projects for those agents
        for subset_index, subset_agents in enumerate(combinations):
            for i in subset_agents:
                m.addConstr(z[i] >= y[subset_index], name="y_"+str(subset_index)+"_"+str(i))

        m.write("wowa.lp")

        # Optimize model
        m.optimize()

        print("X: ", x.X)
        print("Y: ", y.X)
        print("Z: ", z.X)
        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return z.X, m.Runtime