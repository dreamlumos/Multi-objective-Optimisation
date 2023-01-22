import numpy as np
import itertools
import gurobipy as gp
from gurobipy import GRB, quicksum


# -------- Choquet LP -------- #

def choquet_lp(n, p, utilities, costs, v):
    """
    :param n: number of objectives
    :param p: number of projects
    :param utilities: U
    :param costs: costs for each project: [c1, ..., ck] with k in {1, ..., p}
    :param v: capacité (fonction de croyance)

    :type n: int
    :type p: int
    :type utilities: ndarray[int]
    :type costs: ndarray[int]

    :return solution: x
    :rtype: ndarray[int]
    """

    try:
        # Create a new model
        m = gp.Model("Choquet")

        list_projet = [i for i in range(p)]

        combinations = []
        for i in range(len(list_projet) + 1):
            combinations.extend(list(itertools.combinations(list_projet, i)))

        # Create binary variables x_1, ..., x_p
        x = m.addVars(p, vtype=GRB.BINARY, name="x")

        # ----- Contraintes -----

        # le coût total des projets sélectionnés ne dépasse pas l'enveloppe budgétaire fixée
        b = sum(costs)/2  # l'enveloppe budgétaire
        m.addConstr(quicksum(costs[i]*x[i] for i in range(p)) <= b, name="budget")

        # z[i][x] : aptitude d'un ensemble de projets x à satisfaire l'objectif i
        # est définie comme la somme des utilités uij des projets j sélectionnés
        z = {}

        for i in range(n):
            z[i] = {}

        for k, ens_proj in enumerate(combinations):
            for i in range(n):
                if ens_proj:
                    somme = 0
                    for j in ens_proj:
                        somme += utilities[i][j] * x[j]
                    z[i][ens_proj] = somme
                else:
                    z[i][ens_proj] = 0

        # Set objective
        # m.setObjective(quicksum(v[i]*z[i] for i in range(n)), GRB.MAXIMIZE)



        m.write("choquet.lp")

        # Optimize model
        m.optimize()

        print("X: ", x.X)
        print("Z: ", z.X)
        print("B: ", b.X)
        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return x, m.Runtime