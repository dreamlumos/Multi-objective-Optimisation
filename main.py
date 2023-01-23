import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from statistics import mean
import random
import numpy as np

from OWA import *
from Choquet import *
from utils import *


def solve_OWA_problem(filepath=None, nb_agents=None, alpha=None, one_to_one=True, verbose=False):
    """
    Solve a multi-objective problem using OWA. 
    Parameters that are not given will be generated randomly.
    """

    if filepath != None:
        nb_agents, nb_items, utilities = parse_OWA_problem(filepath)
        if nb_agents != nb_items:
            print("Error: nb_agents must be equal to nb_items.")
            return
    else:
        if nb_agents == None:
            nb_agents = random.randint(0, 10)
            nb_items = nb_agents
        utilities = generate_OWA_problem(nb_agents, nb_items)
        print(utilities)
    # weights = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    weights = OWA_weights_generator(nb_agents, alpha)
    solution, runtime = OWA_LP(nb_agents, nb_items, utilities, weights, one_to_one)

    if verbose:
        print("____________________________")
        print("Utilities:", utilities)
        print("Solution:", solution)

        plt.title("Satisfaction of each agent")
        plt.bar(["Agent " + str(i) for i in range(nb_agents)], solution)
        plt.show()

        plt.title("Lorenz vector of the OWA solution")
        plt.bar(["Component " + str(i) for i in range(nb_agents)], lorenz_vector(solution))
        plt.show()


def question_1_1(alpha_min=1, alpha_max=10, plot_figures=False):
    """
    Analysis of solutions for the given example depending on the value of alpha.
    """

    print("Question 1.1: Analysis of solutions for the given example depending on the value of alpha.")

    filepath = "owa_example.txt"
    nb_agents, nb_items, utilities = parse_OWA_problem(filepath)

    # Bar chart configurations
    width = 1 / 14
    x = [i + 0.5 for i in range(nb_agents)]
    fig1, ax1 = plt.subplots(figsize=(6.6, 4))
    fig2, ax2 = plt.subplots(figsize=(6.6, 4))
    cmap = mpl.cm.get_cmap('Blues')
    norm = mpl.colors.Normalize(vmin=alpha_min, vmax=alpha_max)
    colours = [cmap(i) for i in np.linspace(0.1, 1, num=11)]

    runtimes = []
    alpha_list = range(alpha_min, alpha_max + 1, (alpha_max + 1 - alpha_min) // 10)
    print(alpha_list)
    # Experiments
    for exp in range(len(alpha_list)):
        alpha = alpha_list[exp]
        weights = OWA_weights_generator(nb_agents, alpha)
        solution, runtime = OWA_LP(nb_agents, nb_items, utilities, weights, one_to_one=True)
        runtimes.append(runtime)
        ax1.bar([i + width + exp * (1 / 12) for i in range(nb_agents)], solution, width=width, color=colours[exp],
                label="alpha = " + str(alpha))
        ax2.bar([i + width + exp * (1 / 12) for i in range(nb_agents)], lorenz_vector(solution), width=width,
                color=colours[exp], label="alpha = " + str(alpha))

    print("____________________________")
    print("Utilities:", utilities)
    print("Solution:", solution)

    if plot_figures:
        plt.figure(1)
        ax1.title.set_text("Satisfaction of each agent")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["Agent " + str(i) for i in range(1, nb_agents + 1)])
        ax1.set_yticks(range(0, 21, 2))

        fig1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1], label='alpha')
        plt.savefig("question_1_1_solution.png")

        plt.figure(2)
        ax2.title.set_text("Lorenz components of the OWA solutions")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["L" + str(i) for i in range(nb_agents)])
        fig2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax2], label='alpha')
        plt.savefig("question_1_1_Lorenz.png")

        plt.figure(3)
        plt.plot(alpha_list, runtimes)
        plt.savefig("question_1_1_runtimes.png")
        plt.show()

    """
    Analysis of execution time for OWA problems of various sizes.
    """

    avg_times = []  # average execution time for each pair (n,p)
    for nb_agents in nb_agents_list:
        nb_items = 5 * nb_agents
        times = []
        for i in range(10):
            utilities = generate_OWA_problem(nb_agents, nb_items)
            weights = OWA_weights_generator(nb_agents)
            solution, runtime = OWA_LP(nb_agents, nb_items, utilities, weights, one_to_one=one_to_one)
            times.append(runtime)
        avg_times.append(np.mean(times))

    np.savetxt("question_1_2_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", avg_times)
    plt.title("Average execution times for OWA problems of various sizes")
    plt.xlabel("Size in number of agents n (with nb_items = 5*n)")
    plt.ylabel("Average Gurobi Runtime for 10 instances (seconds)")
    plt.plot(nb_agents_list, avg_times)
    plt.savefig("question_1_2_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
    plt.show()


def question_1_3(alpha_list=[2, 5]):
    """
    Analysis of the evolution of solutions when the p vector is varied for the values of alpha provided.
    """

    filepath = "owa_example.txt"
    nb_agents, nb_items, utilities = parse_OWA_problem(filepath)

    p_list = []
    equality_component = 1 / nb_agents
    for extremum_i in range(nb_agents):  # for each extremum
        for step in range(1, nb_agents):  # nb of steps
            for component_i in range(nb_agents):  # for each component of the vector
                new_p = [0] * nb_agents
                if component_i == extremum_i:
                    new_p[component_i] = step * equality_component
                else:
                    new_p[component_i] = (1 - (step * equality_component)) / (nb_agents - 1)
                p_list.append(new_p)

    for alpha in alpha_list:
        for p in p_list:
            # TODO: WOWA LP
            print("TODO: WOWA LP")


def question_1_4(nb_agents_list=[5, 10, 15]):
    """
    Analysis of execution time for WOWA problems of various sizes.
    """

    avg_times = []  # average execution time for each pair (n,p)
    for nb_agents in nb_agents_list:
        nb_items = 5 * nb_agents
        times = []
        for i in range(10):
            utilities = generate_OWA_problem(nb_agents, nb_items)
            # TODO: p generator
            # TODO: WOWA LP
            # times.append(runtime)
        # avg_times.append(np.mean(times))

    # np.savetxt("question_1_4_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", avg_times)
    # plt.title("Average execution times for WOWA problems of various sizes")
    # plt.xlabel("Size in number of agents n (with nb_items = 5*n)")
    # plt.ylabel("Gurobi Runtime (seconds)")
    # plt.plot(nb_agents_list, avg_times)
    # plt.savefig("question_1_4_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".png")
    # plt.show()


def question_2_2(nb_tests=10):
    """
    Analysis of some solutions found for the given Choquet example using the Choquet integral.
    """

    filepath = "choquet_example.txt"
    nb_objectives, nb_projects, utilities, costs = parse_Choquet_problem(filepath)

    #max_mean_mobius_masses = []
    #max_mean_solution, _ = choquet_lp(nb_objectives, nb_projects, utilities, max_mean_mobius_masses)

    all_mobius_masses = []
    all_solutions = []
    all_times = []
    for i in range(nb_tests):
        mobius_masses = belief_function_generator(nb_projects)
        solution, time = choquet_lp(nb_objectives, nb_projects, costs, utilities, mobius_masses)
        all_mobius_masses.append(mobius_masses)
        all_solutions.append(solution)
        all_times.append(time)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print("____________ Question 2.2 ____________")
    for i in range(nb_tests):
        print("\n___ Test "+str(i+1)+" of "+str(nb_tests)+" ___")
        print("  Mobius masses: ", all_mobius_masses[i])
        print("  Solution: ", all_solutions[i])
        print("  Score on objective 1: ", np.sum(utilities[0] * all_solutions[i]))
        print("  Score on objective 2: ", np.sum(utilities[1] * all_solutions[i]))
    
    print("\nMean execution time: ", mean(all_times))


def question_2_3():
    list_n = [2, 5, 10]  # liste des nombres d'objectifs
    list_p = [5, 10, 15, 20]  # liste des nombres de projets
    num_matrices = 10  # nombre de matrices à générer aléatoirement
    dict_mean_time = {}  # dictionnaire du temps moyen d'exécution pour un couple (n,p)

    f = open("question_2_3.txt", "w")

    print("========== (2.3) START ==========")

    for n in list_n:
        for p in list_p:
            list_times = []  # liste du temps d'exécution des matrices U

            # générer toutes les combinaisons de projets possibles
            projects_list = [i for i in range(p)]
            all_combinations = powerset(projects_list)

            for i in range(num_matrices):
                print(f"---------- n={n} p={p} u={i} ----------")
                # matrice U de taille (n, p) avec des coefficients aléatoire entre 1 et 20
                utilities = np.random.randint(1, 21, size=(n, p))

                # liste de taille p des couts des projets tirés aléatoirement entre 10 et 100
                costs = np.random.randint(10, 101, size=p)

                # générer les masses de mobius
                mobius_masses = belief_function_generator(p)

                # optimisation de l'intégrale de choquet
                solution, time = choquet_lp(n, p, costs, utilities, mobius_masses, liste_combinaisons=all_combinations)

                # ajout du temps d'exécution dans la liste
                list_times.append(time)

            # ajout de la moyenne du temps d'exécution pour le couple (n, p)
            dict_mean_time[(n, p)] = mean(list_times)

            # enregistrer la valeur dans un fichier
            f.write(str(n) + "," + str(p) + "," + str(dict_mean_time[(n, p)]) + "\n")
            print(f"Temps d'exécution moyen pour ({n}, {p}) : {dict_mean_time[(n, p)]}")

    print("========== (2.3) END ==========")

    # affichage des temps moyens d'exécution pour les couples (n, p)
    print("Temps d'exécution moyen : ")
    for n in list_n:
        for p in list_p:
            print(f"({n}, {p}) : {dict_mean_time[(n, p)]}")

    f.close()


# -------- Main -------- #

if __name__ == "__main__":
    print("Multi-objective Optimisation")

    seed = 0
    random.seed(seed)

    # solve_OWA_problem("owa_example.txt", alpha=1, verbose=True)
    # solve_OWA_problem()
    # question_1_1(alpha_max=10, plot_figures=True)
    # question_1_2([i for i in range(3, 15)])
    # question_1_2(one_to_one=False)
    # question_1_3()

    # question_2_2(10)
    question_2_3()
