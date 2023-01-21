from OWA import *
from utils import *


if __name__ == "__main__":

	print("Multi-objective Optimisation")

	filepath = "given_example.txt"
	alpha = 1

	nb_agents, nb_items, utilities = parse_OWA_problem(filepath)

	#weights = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
	weights = OWA_weights_generator(alpha, nb_agents)
	solution = OWA_LP(nb_agents, nb_items, utilities, weights)