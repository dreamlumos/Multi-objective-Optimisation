import numpy as np

# -------- UTILS -------- #

def read_str(file): return file.readline().strip()

def read_int(file): return int(read_str(file))

def read_ints(file): return list(map(int, read_str(file).split()))

def parse_OWA_problem(filepath):

	with open(filepath) as file:
		nb_items = read_int(file)
		nb_agents = read_int(file)
		utilities = []

		for i in range(nb_agents):
			utilities.append(read_ints(file))

		utilities = np.array(utilities)
		print(utilities)

	return nb_items, nb_agents, utilities

def OWA_weights_generator(alpha, n):

	if alpha < 1:
		print("Error: alpha must be greater than or equals to 1.") 

	weights = [((n-i+1)/2)**alpha - ((n-i)/n)**alpha for i in range(n)]
	return np.array(weights)