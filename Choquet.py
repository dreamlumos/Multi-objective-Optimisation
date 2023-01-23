import numpy as np
import itertools
import random
import gurobipy as gp
from gurobipy import GRB, quicksum


# -------- Choquet LP -------- #

def choquet_lp(n, p, costs, utilities):
    """
    :param n: number of objectives
    :param p: number of projects
    :param costs: costs for each project: [c1, ..., ck] with k in {1, ..., p}
    :param utilities: U

    :type n: int
    :type p: int
    :type costs: ndarray[int]
    :type utilities: ndarray[int]

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

        # s_i variable binaire pour si le projet i est sélectionné ou pas
        s = m.addMVar(shape=p, vtype=GRB.BINARY, name="x")

        # le coût total des projets sélectionnés ne dépasse pas l'enveloppe budgétaire fixée
        b = sum(costs) / 2  # l'enveloppe budgétaire
        m.addConstrs(quicksum(costs[i] * s[i] for i in range(p)) <= b, name="budget")

        # z[i][x] : aptitude d'un ensemble de projets x à satisfaire l'objectif i
        # est définie comme la somme des utilités uij des projets j sélectionnés
        z = []
        for i in range(n):
            z.append([])
            for x in combinaisons:
                if len(x) == 0:
                    z[i].append(0)
                else:
                    somme = 0
                    for j in x:
                        somme += utilities[i][j] * s[j]
                    z[i].append(somme)
        z = np.array(z)

        y = m.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="y")
        m.addConstrs((y[i] < z[i] for i in range(n)), name="aptitude")

        # calculer les masses de mobius
        mobius = []
        for i in range(n):
            mobius.append([])
            for x in combinaisons:
                if len(x) == 0:
                    mobius[i].append(0)
                elif len(x) == p:
                    mobius[i].append(1)
                else:
                    epsilon = 1e-9  # pour éviter d'avoir 0 ou 1
                    mobius[i].append(random.uniform(0 + epsilon, 1 - epsilon))
        mobius = np.array(mobius)

        # Set objective
        # m.setObjective(quicksum(v[i]*z[i] for i in range(n)), GRB.MAXIMIZE)

        m.write("choquet.lp")

        # Optimize model
        m.optimize()

        print("Y: ", y.X)
        print("Z: ", z.X)
        print("S: ", s.X)
        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return y, m.Runtime
