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

        # variable binaire pour savoir quel projet est sélectionné
        selection = m.addMVar(shape=p, vtype=GRB.BINARY, name="x")

        # le coût total des projets sélectionnés ne dépasse pas l'enveloppe budgétaire fixée
        b = sum(costs) / 2  # l'enveloppe budgétaire
        m.addConstr(quicksum(costs[i] * selection[i] for i in range(p)) <= b, name="budget")

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
                        somme += utilities[i][j] * selection[j]
                    z[i].append(somme)
        z = np.array(z)

        y = m.addMVar(shape=len(combinaisons), vtype=GRB.CONTINUOUS, name="y")
        for j, x in enumerate(combinaisons):
            m.addConstr(quicksum(z[i][j] for i in range(n)) >= y[j], name="aptitude_"+str(x))

        # calculer les masses de mobius
        mobius = np.random.dirichlet([1 for j in range(len(combinaisons))])

        # Set objective
        m.setObjective(mobius @ y, GRB.MAXIMIZE)

        m.write("choquet.lp")

        # Optimize model
        m.optimize()

        # Récupérer les solutions
        max_value = max(y.X)
        indices = [i for i, x in enumerate(y.X) if x == max_value]
        solutions = [combinaisons[i] for i in indices]

        print("Y: ", y.X)
        print("Solutions: ", solutions)
        print("Selection: ", selection.X)
        # print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return solutions, m.Runtime
