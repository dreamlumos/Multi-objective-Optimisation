import numpy as np

import gurobipy as gp
from gurobipy import GRB

# -------- OWA LP -------- #

def OWA_LP(n, p, utilities, weights, one_to_one=True):
    """
    :param n: nb_agents
    :param p: nb_items
    :param utilities: U
    :param weights: [w_1, w_2, ..., w_n] in order of increasing ordered components (decreasing weights)
    :param one_to_one: indicates whether only one item is to be attributed per agent

    :type nb_agents: int
    :type nb_items: int
    :type utilities: ndarray[int]
    :type weights: ndarray[int]
    :type one_to_one: bool

    :return solution: x
    :rtype: ndarray[int]
    """

    try:

        # Create a new model
        m = gp.Model("OWA")

        # Create variables y_1, y_2, ..., y_n
        y = m.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="y")

        # Set objective
        obj = weights @ y
        m.setObjective(obj, GRB.MAXIMIZE)

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

        #### OWA and linearisation constraints ####

        # Impose order of y_i variables (y_1 <= y_2 <= ... <= y_n)
        for i in range(1, n):
            m.addConstr(y[i-1] <= y[i], name="c_y_"+str(i))

        # Calculate value of M to use (has to be larger than any value y_i or z_i could take)
        M = np.sum(utilities) * 10 

        # Constraints that associate z_i and y_i variables
        b = m.addMVar(shape=(n,n), vtype=GRB.BINARY, name="b")
        m.addConstrs((y[k] * np.ones(n) <= z + M * b[k,:] for k in range(n)), name="c_yz")
        m.addConstrs((b[k,:] @ np.ones(n) == k for k in range(n)), name="c_b")

        m.write("owa.lp")

        # Optimize model
        m.optimize()

        print("X: ", x.X)
        print("Y: ", y.X)
        print("Z: ", z.X)
        print("B: ", b.X)
        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return z.X, m.Runtime