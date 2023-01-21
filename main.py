import datetime
import matplotlib.pyplot as plt
import random

from OWA import *
from utils import *

def solve_OWA_problem(filepath=None, nb_agents=None, alpha=None):
	"""
	Solve a multi-objective problem using OWA. 
	Note: only cases where nb_agents == nb_items are considered for the moment.
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

	#weights = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
	weights = OWA_weights_generator(nb_agents, alpha)
	solution, _ = OWA_LP(nb_agents, nb_items, utilities, weights)

def question_1_1():
	"""
	Analysis of solutions for the given example depending on the value of alpha.
	"""

	print("Question 1.1 : Analysis of solutions for the given example depending on the value of alpha.")
	
	filepath = "given_example.txt"
	nb_agents, nb_items, utilities = parse_OWA_problem(filepath)

	for alpha in range(1, 10):
		weights = OWA_weights_generator(nb_agents, alpha)
		solution, _ = OWA_LP(nb_agents, nb_items, utilities, weights, one_one=True)

	# TODO: generate histograms

def question_1_2():
	"""
	Analysis of execution time for OWA problems of various sizes.
	"""

	avg_times = [] # average execution time for each pair (n,p)
	for nb_agents in [5, 10, 15]:
		nb_items = 5 * nb_agents
		times = []
		for i in range(10):
			utilities = generate_OWA_problem(nb_agents, nb_items)
			weights = OWA_weights_generator(nb_agents)
			solution, runtime = OWA_LP(nb_agents, nb_items, utilities, weights, one_one=False)
			times.append(runtime)
		avg_times.append(np.mean(times))

	np.savetxt("question_1_2_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", avg_times)
	plt.plot(avg_times)
	plt.savefig("question_1_2_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".png")
	plt.show()

if __name__ == "__main__":

	print("Multi-objective Optimisation")

	seed = 0
	random.seed(seed)

	#solve_OWA_problem("given_example.txt", alpha=1)
	#question_1_1()
	question_1_2()