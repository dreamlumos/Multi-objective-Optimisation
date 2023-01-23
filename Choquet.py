import numpy as np
import itertools
import random
import gurobipy as gp
from gurobipy import GRB, quicksum


# -------- Choquet LP -------- #

def choquet_lp(n, p, costs, utilities, mobius_masses=None):
    """
    :param n: number of objectives
    :param p: number of projects
    :param costs: costs for each project: [c1, ..., ck] with k in {1, ..., p}
    :param utilities: U
    :param mobius_masses: Mobius masses

    :type n: int
    :type p: int
    :type costs: ndarray[int]
    :type utilities: ndarray[int]
    :type mobius_masses: ndarray[float]

    :return solution: x
    :rtype: ndarray[int]
    """

    try:
        # Create a new model
        m = gp.Model("Choquet")

        # Avoir toutes les combinaisons de projets possibles
        liste_projets = [i for i in range(p)]
        combinaisons = []
        for i in range(len(liste_projets) + 1):
            combinaisons.extend(list(itertools.combinations(liste_projets, i)))

        # variable binaire pour savoir quel projet est sélectionné
        x = m.addMVar(shape=p, vtype=GRB.BINARY, name="x")

        # le coût total des projets sélectionnés ne dépasse pas l'enveloppe budgétaire fixée
        b = sum(costs) / 2  # l'enveloppe budgétaire
        m.addConstr(quicksum(costs[i] * x[i] for i in range(p)) <= b, name="budget")

        y = m.addMVar(shape=len(combinaisons), vtype=GRB.CONTINUOUS, name="y")
        for k, ss in enumerate(combinaisons):
            for i in range(n):
                # contrainte sur l'aptitude d'un ensemble de projets ss z_i(ss)
                # à satisfaire l'objectif i est définie comme la somme des utilités uij des projets j sélectionnés
                m.addConstr(quicksum(utilities[i][j] * x[j] for j in ss) >= y[k], name="aptitude_"+str(i)+"_"+str(k))

        # calculer les masses de mobius
        if mobius_masses:
            mobius = mobius_masses
        else:
            mobius = np.random.dirichlet([1 for j in range(len(combinaisons))])

        # Set objective
        m.setObjective(mobius @ y, GRB.MAXIMIZE)

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
