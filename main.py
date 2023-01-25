import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from statistics import mean
import random
import numpy as np

from OWA import *
from WOWA import *
from Choquet import *
from Choquet_graph import *
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

def question_1_2(nb_agents_list=[5, 10, 15], one_to_one=True):
    """
    Analysis of execution time for OWA problems of various sizes.
    """

    avg_times = [] # average execution time for each pair (n,p)
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


def question_1_3(alpha_list=[2, 5], plot_figures=False):
    """
    Analysis of the evolution of solutions when the p vector is varied for the values of alpha provided.
    """

    filepath = "owa_example.txt"
    nb_agents, nb_items, utilities = parse_OWA_problem(filepath)

    # Bar chart configurations
    width = 1 / 8
    x = [i + 0.5 for i in range(nb_agents)]
    figures = [None]*len(alpha_list)
    axes = [None]*len(alpha_list)
    for i in range(len(alpha_list)):
        figures[i], axes[i] = plt.subplots(5, figsize=(6.6, 6))
    cmap = mpl.cm.get_cmap('Reds')
    norm = mpl.colors.Normalize(vmin=1, vmax=5)
    colours = [cmap(i) for i in np.linspace(0.1, 1, num=6)]

    p_list = []
    equality_component = 1 / nb_agents
    for extremum_i in range(nb_agents):  # for each extremum
        p_sublist = []
        for step in range(1, nb_agents+1):  # nb of steps
            new_p = [0] * nb_agents
            for component_i in range(nb_agents):  # for each component of the vector
                if component_i == extremum_i:
                    new_p[component_i] = step * equality_component
                else:
                    new_p[component_i] = (1 - (step * equality_component)) / (nb_agents - 1)
            p_sublist.append(new_p)
        p_list.append(p_sublist)

    for alpha_i in range(len(alpha_list)):
        alpha = alpha_list[alpha_i]
        for extremum_i in range(nb_agents):
            for exp in range(nb_agents):
                p = p_list[extremum_i][exp]
                mobius_masses = WOWA_mobius_mass_generator(p, alpha)
                solution, runtime = WOWA_LP(nb_agents, nb_items, utilities, mobius_masses, one_to_one=True)
                axes[alpha_i][extremum_i].bar([i + width + exp * (1 / 6) for i in range(nb_agents)], solution, width=width, color=colours[exp])
                print("_______")
                print("p:", p)
                print("mobius: ", mobius_masses)

    print("____________________________")
    print("Utilities:", utilities)

    if plot_figures:
        for alpha_i in range(len(alpha_list)):
            plt.figure(alpha_i+1)
            for extremum_i in range(nb_agents):
                # axes[alpha_i][extremum_i].title.set_text("Satisfaction of each agent")
                axes[alpha_i][extremum_i].set_xticks(x)
                axes[alpha_i][extremum_i].set_xticklabels(["Agent " + str(i) for i in range(1, nb_agents + 1)])
                axes[alpha_i][extremum_i].set_yticks(range(0, 21, 5))
                axes[alpha_i][extremum_i].label_outer()

                figures[alpha_i].colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[axes[alpha_i][extremum_i]], label='step')
            plt.savefig("question_1_3_plot_"+str(alpha_i)+".png")
        plt.show()


def question_1_4(nb_agents_list=[5, 10, 15]):
    """
    Analysis of execution time for WOWA problems of various sizes.
    """

    avg_times = [] # average execution time for each pair (n,p)
    for nb_agents in nb_agents_list:
        nb_items = 5 * nb_agents
        times = []
        for i in range(10):
            utilities = generate_OWA_problem(nb_agents, nb_items)
            p = WOWA_importance_weights_generator(nb_agents)
            alpha = random.randint(1, 10)
            mobius_masses = WOWA_mobius_mass_generator(p, alpha)
            print("Utilities: ", utilities)
            print("p: ", p)
            print("alpha: ", alpha)
            print("mobius_masses: ", mobius_masses)
            solution, runtime = WOWA_LP(nb_agents, nb_items, utilities, mobius_masses, one_to_one=True)
            times.append(runtime)
        avg_times.append(np.mean(times))

    np.savetxt("question_1_4_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", avg_times)
    plt.title("Average execution times for WOWA problems of various sizes")
    plt.xlabel("Size in number of agents n (with nb_items = 5*n)")
    plt.ylabel("Gurobi Runtime (seconds)")
    plt.plot(nb_agents_list, avg_times)
    plt.savefig("question_1_4_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".png")
    plt.show()


def question_2_2(nb_tests=10):
    """
    Analysis of some solutions found for the given Choquet example using the Choquet integral.
    """

    filepath = "choquet_example.txt"
    nb_objectives, nb_projects, utilities, costs = parse_Choquet_problem(filepath)

    all_mobius_masses = []
    all_solutions = []
    all_times = []
    for i in range(nb_tests):
        mobius_masses = belief_function_generator(nb_objectives)
        solution, time = choquet_lp(nb_objectives, nb_projects, costs, utilities, mobius_masses)
        all_mobius_masses.append(mobius_masses)
        all_solutions.append(solution)
        all_times.append(time)

    max_mean_mobius_masses = np.array([0, 0.5, 0.5, 0])
    max_mean_solution, time = choquet_lp(nb_objectives, nb_projects, costs, utilities, mobius_masses)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print("____________ Question 2.2 ____________")

    print("___ Solution maximising mean satisfaction ___")
    print("  Mobius masses: ", max_mean_mobius_masses)
    print("  Solution: ", max_mean_solution)
    print("  Score on objective 1: ", np.sum(utilities[0] * max_mean_solution))
    print("  Score on objective 2: ", np.sum(utilities[1] * max_mean_solution))

    for i in range(nb_tests):
        print("\n___ Test "+str(i+1)+" of "+str(nb_tests)+" ___")
        print("  Mobius masses: ", all_mobius_masses[i])
        print("  Solution: ", all_solutions[i])
        print("  Score on objective 1: ", np.sum(utilities[0] * all_solutions[i]))
        print("  Score on objective 2: ", np.sum(utilities[1] * all_solutions[i]))
    
    print("\nMean execution time: ", mean(all_times))


def question_2_3(n_list=[2, 5, 10], p_list=[5, 10, 15, 20]):
    """
    Analysis of execution time for Choquet problems of various sizes.

    :param n_list: list of nb_objectives to test
    :param p_list: list of nb_projects to test
    
    :type n_list: list[int]
    :type p_list: list[int]
    """
    
    nb_instances = 10  # nombre de matrices à générer aléatoirement
    dict_mean_time = {}  # dictionnaire du temps moyen d'exécution pour un couple (n,p)

    f = open("question_2_3.txt", "w")

    print("========== (2.3) START ==========")

    for n in n_list:
        for p in p_list:
            list_times = []  # liste des temps d'exécution pour les instances de taille (n, p)

            # générer toutes les combinaisons de projets possibles
            objectives_list = [i for i in range(n)]
            combinations = powerset(objectives_list)

            for i in range(nb_instances):
                print(f"---------- n={n} p={p} u={i} ----------")
                
                utilities, costs, mobius_masses = generate_Choquet_problem(n, p)

                # optimisation de l'intégrale de choquet
                solution, time = choquet_lp(n, p, costs, utilities, mobius_masses, combinations)

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
    for n in n_list:
        for p in p_list:
            print(f"({n}, {p}) : {dict_mean_time[(n, p)]}")

    f.close()

    plt.title("Average execution times for Choquet problems of various sizes")
    plt.xlabel("Size in number of projects p")
    plt.ylabel("Average Gurobi Runtime for 10 instances (seconds)")
    for n in n_list:
        plt.plot(p_list, [dict_mean_time[(n, p)] for p in p_list])
    plt.legend([str(i)+" objectives" for i in n_list])
    plt.savefig("question_2_3_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_obj.png")
    plt.show()
    plt.clf()

    plt.title("Average execution times for Choquet problems of various sizes")
    plt.xlabel("Size in number of objectives n")
    plt.ylabel("Average Gurobi Runtime for 10 instances (seconds)")
    for p in p_list:
        plt.plot(n_list, [dict_mean_time[(n, p)] for n in n_list])
    plt.legend([str(i) + " projects" for i in p_list])
    plt.savefig("question_2_3_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_proj.png")
    plt.show()


def plot_question_2_3():
    f = open("question_2_3.txt", "r")
    lines = f.readlines()

    n_list = []
    p_list = []
    runtimes = {}

    for line in lines:
        split_line = line.strip("\n").split(",")
        n = int(split_line[0])
        p = int(split_line[1])
        time = float(split_line[2])
        if n not in n_list:
            n_list.append(n)
        if p not in p_list:
            p_list.append(p)
        runtimes[(n, p)] = time

    plt.title("Average execution times for Choquet problems of various sizes")
    plt.xlabel("Size in number of objectives n")
    plt.ylabel("Average Gurobi Runtime for 10 instances (seconds)")
    for p in p_list:
        plt.plot(n_list, [runtimes[(n, p)] for n in n_list])
    plt.legend([str(i) + " projects" for i in p_list])
    plt.savefig("question_2_3_times_size_projects.png")
    plt.clf()

    plt.title("Average execution times for Choquet problems of various sizes")
    plt.xlabel("Size in number of projects p")
    plt.ylabel("Average Gurobi Runtime for 10 instances (seconds)")
    for n in n_list:
        plt.plot(p_list, [runtimes[(n, p)] for p in p_list])
    plt.legend([str(i) + " objectives" for i in n_list])
    plt.savefig("question_2_3_times_size_objectives.png")

    f.close()

def question_graph():

    n = 2 # Number of scenarios (number of objectives)

    traveling_time = [[[0, 5, 10, 2, 999, 999, 999],
                        [999, 0, 4, 1, 4, 999, 999],
                        [999, 999, 0, 999, 3, 1, 0],
                        [999, 999, 1, 0, 3, 3, 999],
                        [999, 999, 999, 999, 0, 999, 1],
                        [999, 999, 999, 999, 999, 0, 1],
                        [999, 999, 999, 999, 999, 999, 0]],
                    [[0, 3, 4, 6, 999, 999, 999],
                        [999, 0, 2, 3, 6, 999, 999],
                        [999, 999, 0, 999, 1, 1, 0],
                        [999, 999, 4, 0, 999, 5, 999],
                        [999, 999, 999, 999, 0, 999, 1],
                        [999, 999, 999, 999, 999, 0, 1],
                        [999, 999, 999, 999, 999, 999, 0]]]

    mobius_masses = np.array([0, 1/3, 1/3, 1/3])

    solution, runtime = choquet_graph_lp(n, np.array(traveling_time), mobius_masses)

    print(solution)

# -------- Main -------- #

if __name__ == "__main__":
    print("Multi-objective Optimisation")

    seed = 0
    random.seed(seed)

    # solve_OWA_problem("owa_example.txt", alpha=1, verbose=True)
    # solve_OWA_problem()
    # question_1_1(alpha_max=10, plot_figures=True)
    # question_1_2([i for i in range(3, 30)])
    # question_1_2(one_to_one=False)
    # question_1_3(plot_figures=True)
    # question_1_4()

    # question_2_2(10)
    # question_2_3()
    # plot_question_2_3()

    # question_graph()