import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

from OWA import *
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
    #weights = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    weights = OWA_weights_generator(nb_agents, alpha)
    solution, runtime = OWA_LP(nb_agents, nb_items, utilities, weights, one_to_one)

    if verbose:
        print("____________________________")
        print("Utilities:", utilities)
        print("Solution:", solution)

        plt.title("Satisfaction of each agent")
        plt.bar(["Agent "+str(i) for i in range(nb_agents)], solution)
        plt.show()

        plt.title("Lorenz vector of the OWA solution")
        plt.bar(["Component "+str(i) for i in range(nb_agents)], lorenz_vector(solution))
        plt.show()

def question_1_1(plot_figures=False):
    """
    Analysis of solutions for the given example depending on the value of alpha.
    """

    print("Question 1.1 : Analysis of solutions for the given example depending on the value of alpha.")
    
    filepath = "owa_example.txt"
    nb_agents, nb_items, utilities = parse_OWA_problem(filepath)

    # Bar chart configurations
    width = 1/14
    x = [i + 0.5 for i in range(nb_agents)]
    fig1, ax1 = plt.subplots(figsize=(6.6, 4))
    fig2, ax2 = plt.subplots(figsize=(6.6, 4))
    cmap = mpl.cm.get_cmap('Blues')
    norm = mpl.colors.Normalize(vmin=1, vmax=10)
    colours = [cmap(i) for i in np.linspace(0.1, 1, num=11)]

    runtimes = []
    # Experiments
    for alpha in range(1, 11):
        weights = OWA_weights_generator(nb_agents, alpha)
        solution, runtime = OWA_LP(nb_agents, nb_items, utilities, weights, one_to_one=True)
        runtimes.append(runtime)
        ax1.bar([i + width + alpha * (1/12) for i in range(nb_agents)], solution, width=width, color=colours[alpha], label="alpha = "+str(alpha))
        ax2.bar([i + width + alpha * (1/12) for i in range(nb_agents)], lorenz_vector(solution), width=width, color=colours[alpha], label="alpha = "+str(alpha))

    print("____________________________")
    print("Utilities:", utilities)
    print("Solution:", solution)

    if plot_figures:
        plt.figure(1)
        ax1.title.set_text("Satisfaction of each agent")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["Agent "+str(i) for i in range(1, nb_agents+1)])
        fig1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1], label='alpha')
        plt.savefig("question_1_1_solution.png")

        plt.figure(2)
        ax2.title.set_text("Lorenz components of the OWA solutions")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["L"+str(i) for i in range(nb_agents)])
        fig2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax2], label='alpha')
        plt.savefig("question_1_1_Lorenz.png")

        plt.figure(3)
        plt.plot([i for i in range(1, 11)], runtimes)
        plt.savefig("question_1_1_runtimes.png")
        plt.show()

def question_1_2(nb_agents_list=[5, 10, 15]):
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
            solution, runtime = OWA_LP(nb_agents, nb_items, utilities, weights, one_to_one=True)
            times.append(runtime)
        avg_times.append(np.mean(times))

    np.savetxt("question_1_2_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", avg_times)
    plt.title("Average execution times for OWA problems of various sizes")
    plt.xlabel("Size in number of agents n (with nb_items = 5*n)")
    plt.ylabel("Average Gurobi Runtime for 10 instances (seconds)")
    plt.plot(nb_agents_list, avg_times)
    plt.savefig("question_1_2_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".png")
    plt.show()


# -------- Main -------- #

if __name__ == "__main__":

    print("Multi-objective Optimisation")

    seed = 0
    random.seed(seed)

    #solve_OWA_problem("owa_example.txt", alpha=1, verbose=True)
    #solve_OWA_problem()
    #question_1_1()
    #question_1_2([i for i in range(3, 15)])
    #question_1_3()