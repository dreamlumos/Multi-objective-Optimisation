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
    :param combinations: liste de toutes les combinaisons de projets possibles

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
        projects_list = [i for i in range(p)]
        combinations = powerset(projects_list)

    try:
        # Create a new model
        m = gp.Model("Choquet")

        # Create y variables that indicate value of a subset
        y = m.addMVar(shape=len(combinations), vtype=GRB.CONTINUOUS, name="y")

        # variables binaires pour savoir quels projets sont sélectionnés
        x = m.addMVar(shape=p, vtype=GRB.BINARY, name="x")

        # Set objective
        m.setObjective(mobius_masses @ y, GRB.MAXIMIZE)

        # le coût total des projets sélectionnés ne doit pas dépasser l'enveloppe budgétaire fixée
        b = sum(costs) / 2  # l'enveloppe budgétaire
        m.addConstr(costs @ x <= b, name="budget")

        # contraintes sur l'aptitude d'un ensemble de projets ss z_i(ss) à satisfaire l'objectif i
        # est définie comme la somme des utilités uij des projets j qui appartiennent à l'ensemble
        for k, ss in enumerate(combinations):
            for i in range(n):
                # print(f"({n}, {p}) : add constraint aptitude_{i}_{k}")
                m.addConstr(quicksum(utilities[i][j] * x[j] for j in ss) >= y[k], name="aptitude_"+str(i)+"_"+str(k))

        m.write("choquet.lp")

        # Optimize model
        m.optimize()

        print("X: ", x.X)
        print("Y: ", y.X)
        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return x.X, m.Runtime
