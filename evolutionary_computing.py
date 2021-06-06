from random import randint, random
from convolutional_neural_network import CNN

POPULATION_SIZE = 50
MAXIMUM_GENERATION = 100
STOP_CONDITION = 30 # Number of generations without improvements
TOURNAMENT_SIZE = 3
MIN_POINTS = 3 # Minium number of points in multipoints crossover
MAX_POINTS = 10 # Maximum number of points in multipoints crossover
MUTATION_RATE = 0.015
ELITE_SIZE = 1
GENE_LENGTH = 69
READ_DATA_FILE = "" # Using file to backup data from Google Colab, run with multiple sessions
# READ_DATA_FILE = "" -> first session, new initialization
WRITE_DATA_FILE = "SAVED_SESSION_DATA.txt" # In new session, upload files containing data from previous ones

# Dictionaries to convert genotype to phenotype
LEARNING_RATE_DICT = {
    0: 1 * 10 ** (-5), 1: 5 * 10 ** (-5),
    2: 1 * 10 ** (-4), 3: 5 * 10 ** (-4),
    4: 1 * 10 ** (-3), 5: 5 * 10 ** (-3),
    6: 1 * 10 ** (-2), 7: 5 * 10 ** (-2),
}

DENSE_TYPE_DICT = {
    0: "recurrent", 1: "LSTM", 2: "GRU", 3: "feed-forward"
}

REGULARIZATION_DICT = {
    0: "l1", 1: "l2", 2: "l1l2", 3: None
}

OPTIMIZER_DICT = {
    0: "", 1: "4",
    2: "1", 3: "5",
    4: "2", 5: "6",
    6: "3", 7: "7",
}

ACTIVATION_DICT = {
    0: "relu",
    1: "linear"
}

def tournament_selection(population):
    selected, max_fitness = None, 0
    while selected is None:
        for i in range(TOURNAMENT_SIZE):
            contestant = population[randint(0, POPULATION_SIZE - 1)]
            if contestant.adjusted_fitness > max_fitness:
                selected = contestant
                max_fitness = contestant.adjusted_fitness
    return selected

def crossover(parent1, parent2):
    points_num = randint(MIN_POINTS, MAX_POINTS)
    points = [0]
    for i in range(points_num):
        points.append(randint(points[-1] + 1, GENE_LENGTH - (points_num - i)))
    points.append(GENE_LENGTH)
    gene1, gene2 = [], []
    for i in range(len(points) - 1):
        if i % 2 == 0:
            gene1 += parent1.gene[points[i]:points[i + 1]]
            gene2 += parent2.gene[points[i]:points[i + 1]]
        else:
            gene1 += parent2.gene[points[i]:points[i + 1]]
            gene2 += parent1.gene[points[i]:points[i + 1]]
    children1 = Individual(gene1)
    children2 = Individual(gene2)
    return children1, children2

def count_same_element(array1, array2):
    return [array1[i] == array2[i] for i in range(len(array1))].count(True)

def get_similarity(individual1, individual2):
    nc1 = individual1.get_convol_layers_num()
    nc2 = individual2.get_convol_layers_num()
    nd1 = individual1.get_dense_layers_num()
    nd2 = individual2.get_dense_layers_num()
    if nc1 == nc2 and nd1 == nd2:
        properies_num = nc1 * 4 + nd1 * 5 + 2
        same_count = 0

        f1, f2 = individual1.get_optimizer(), individual2.get_optimizer()
        if f1 == f2:
            same_count += 1
        n1 = individual1.get_learning_rate()
        n2 = individual1.get_learning_rate()
        if n1 == n2:
            same_count += 1

        ck1 = individual1.get_kernels_num(nc1)
        ck2 = individual2.get_kernels_num(nc2)
        same_count += count_same_element(ck1, ck2)
        cs1 = individual1.get_kernel_sizes(nc1)
        cs2 = individual2.get_kernel_sizes(nc2)
        same_count += count_same_element(cs1, cs2)
        cp1 = individual1.get_pooling(nc1)
        cp2 = individual2.get_pooling(nc2)
        same_count += count_same_element(cp1, cp2)
        ca1 = individual1.get_convol_activation(nc1)
        ca2 = individual2.get_convol_activation(nc2)
        same_count += count_same_element(ca1, ca2)

        dt1 = individual1.get_dense_type(nd1)
        dt2 = individual2.get_dense_type(nd2)
        same_count += count_same_element(dt1, dt2)
        dn1 = individual1.get_neurons_num(nd1)
        dn2 = individual2.get_neurons_num(nd2)
        same_count += count_same_element(dn1, dn2)
        da1 = individual1.get_dense_activation(nd1)
        da2 = individual2.get_dense_activation(nd2)
        same_count += count_same_element(da1, da2)
        dr1 = individual1.get_regularization(nd1)
        dr2 = individual2.get_regularization(nd2)
        same_count += count_same_element(dr1, dr2)
        dd1 = individual1.get_dropout(nd1)
        dd2 = individual2.get_dropout(nd2)
        same_count += count_same_element(dd1, dd2)
        return same_count / properies_num
    else:
        return 0

def binary_to_decimal(bits):
    return int("".join(map(str, bits)), 2)

class Individual(object):
    def __init__(self, *args):
        if not args: # New initialization from start
            self.gene = [randint(0, 1) for i in range(GENE_LENGTH)]
        else:
            self.gene = args[0]

    def evaluate(self):
        components = self.get_components()
        self.fitness = cnn.evaluate(components)

    def mutate(self):
        for i in range(GENE_LENGTH):
            odd = random() <= 0.015
            self.gene[i] = int(not self.gene[i]) if odd else self.gene[i]

    def get_convol_layers_num(self):
        return 1 + binary_to_decimal(self.gene[2:4])

    def get_kernels_num(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[4 + i * 10: 4 + i * 10 + 3]
            result.append(2 ** (binary_to_decimal(binary) + 1))
        return result

    def get_kernel_sizes(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[7 + i * 10: 7 + i * 10 + 3]
            result.append(2 + binary_to_decimal(binary))
        return result

    def get_pooling(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[10 + i * 10: 10 + i * 10 + 3]
            result.append(1 + binary_to_decimal(binary))
        return result

    def get_convol_activation(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[13 + i * 10: 13 + i * 10 + 1]
            result.append(ACTIVATION_DICT[binary_to_decimal(binary)])
        return result

    def get_dense_layers_num(self):
        return 1 + binary_to_decimal([self.gene[44]])

    def get_dense_type(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[45 + i * 9: 45 + i * 9 + 2]
            result.append(DENSE_TYPE_DICT[binary_to_decimal(binary)])
        return result

    def get_neurons_num(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[47 + i * 9: 47 + i * 9 + 3]
            result.append(2 ** (binary_to_decimal(binary) + 3))
        return result

    def get_dense_activation(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[50 + i * 9: 50 + i * 9 + 1]
            result.append(ACTIVATION_DICT[binary_to_decimal(binary)])
        return result

    def get_regularization(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[51 + i * 9: 51 + i * 9 + 2]
            result.append(2 ** (binary_to_decimal(binary) + 1))
        return result

    def get_dropout(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[53 + i * 9: 53 + i * 9 + 1]
            result.append(binary_to_decimal(binary) / 2)
        return result

    def get_optimizer(self):
        result = []
        binary = self.gene[63: 66]
        result.append(OPTIMIZER_DICT[binary_to_decimal(binary)])
        return result

    def get_learning_rate(self):
        result = []
        binary = self.gene[66: 69]
        result.append(LEARNING_RATE_DICT[binary_to_decimal(binary)])
        return result

    def get_components(self):
        dct = {}

        # Convolutional layers
        dct["nc"] = self.get_convol_layers_num()
        dct["ck"] = self.get_kernels_num(dct["nc"])
        dct["cs"] = self.get_kernel_sizes(dct["nc"])
        dct["cp"] = self.get_pooling(dct["nc"])
        dct["ca"] = self.get_convol_activation(dct["nc"])

        # Dense layers
        dct["nd"] = self.get_dense_layers_num()
        dct["dt"] = self.get_dense_type(dct["nd"])
        dct["dn"] = self.get_neurons_num(dct["nd"])
        dct["da"] = self.get_dense_activation(dct["nd"])
        dct["dr"] = self.get_dropout(dct["nd"])
        dct["dd"] = self.get_regularization(dct["nd"])

        # Learning parameters
        dct["n"] = self.get_learning_rate()
        dct["f"] = self.get_optimizer()

        return dct

    def to_string(self):
        return "".join(map(str, self.gene))


class Tracker(object):
    def __init__(self):
        self.generation_count = 1
        self.best_fitness = []
        self.population_history = []
        self.best_individual = None

    def stop_condition(self):
        if self.generation_count >= STOP_CONDITION:
            if self.best_fitness[-1] == self.best_fitness[-30]:
                return True
        return False

    def elitism(self):
        return self.best_individual

    def update_elitism(self, population):
        # self.population_history.append(list(population))
        generation = []
        for individual in population:
            generation.append(" ".join(["".join(map(str, individual.gene)), str(individual.fitness), str(individual.adjusted_fitness)]) + "\n")
        self.population_history.append(generation)
        best, _max = None, 0
        for individual in population:
            if individual.fitness > _max:
                best, _max = individual, individual.fitness
        self.best_individual = best
        self.best_fitness.append(_max)

    def print(self):
        print(self.best_fitness)


class Population(object):
    def __init__(self, *args):
        if not args: # New initialization from start
            self.populace = [Individual() for i in range(POPULATION_SIZE)]
        else: # Create a population from text file
            file_name = args[0]

    def calculate_ajusted_fitness(self):
        for i in range(POPULATION_SIZE):
            similarity_sum = 0
            for j in range(POPULATION_SIZE):
                if j != i:
                    sim = get_similarity(self.populace[i], self.populace[j])
                    similarity_sum += sim
            similarity = 1 - (similarity_sum / (POPULATION_SIZE - 1))
            self.populace[i].adjusted_fitness = self.populace[i].fitness
            self.populace[i].adjusted_fitness *= similarity

    def print(self):
        for individual in self.populace:
            print(individual.to_string(), end = " ")
            print(individual.fitness, individual.adjusted_fitness)

def read_data_from_file(file_name):
    population = None
    tracker = None
    return population, tracker

def save_data_to_file(population, tracker):
    text_lines = []
    text_lines.append("CURRENT POPULATION\n")
    for individual in population.populace:
        text_lines.append(" ".join(["".join(map(str, individual.gene)), str(individual.fitness), str(individual.adjusted_fitness)]) + "\n")
    text_lines.append("TRACKER INFO\n")
    text_lines.append(str(tracker.generation_count) + "\n")
    text_lines.append(" ".join(map(str, tracker.best_fitness)) + "\n")
    best_ind = tracker.best_individual
    text_lines.append(" ".join(["".join(map(str, individual.gene)), str(individual.fitness), str(individual.adjusted_fitness)]) + "\n")
    text_lines.append("POPULATION HISTORY")
    for index, generation in enumerate(tracker.population_history):
        text_lines.append("\tGeneration " + str(index) + "\n")
        for individual in generation:
            text_lines.append(individual)

    with open(WRITE_DATA_FILE, "w") as f:
        f.writelines(text_lines)

cnn = CNN()

if READ_DATA_FILE == "":
    population = Population()

    # Remove all invalid individual (invalid CNN model structure)
    for i in range(POPULATION_SIZE):
        population.populace[i].evaluate()
        while population.populace[i].fitness == 0:
            # print("Re-initialize")
            population.populace[i] = Individual()
            population.populace[i].evaluate()
    population.calculate_ajusted_fitness()

    tracker = Tracker()
    tracker.update_elitism(population.populace)
else:
    population, tracker = read_data_from_file(READ_DATA_FILE)

print("Generation 0")
population.print()

# Population evolution
for i in range(tracker.generation_count, tracker.generation_count + 10):
    print("Generation", i)

    # Create parent pool for mating by tournament selection
    pool = []
    for j in range(POPULATION_SIZE):
        pool.append(tournament_selection(population.populace))

    # Create offsrping for next generation by crossover then mutate
    next_generation = []
    for j in range(0, POPULATION_SIZE, 2):
        parent1, parent2 = pool[j], pool[j + 1]
        children1, children2 = crossover(parent1, parent2)
        children1.mutate()
        children2.mutate()
        next_generation += [children1, children2]
    for individual in next_generation:
        individual.evaluate()

    # Remove the worst individual
    worst, min_fitness = None, 100
    for i in range(POPULATION_SIZE):
        if next_generation[i].fitness < min_fitness:
            worst, min_fitness = i, next_generation[i].fitness
    del next_generation[worst]

    # Add the best individual from the previous generation
    next_generation.append(tracker.elitism())

    population.populace = next_generation
    population.calculate_ajusted_fitness()
    tracker.update_elitism(population.populace)
    tracker.generation_count += 1
    population.print()

    # Got 30 generations without improvements
    if tracker.stop_condition():
        break

save_data_to_file(population, tracker)
tracker.print()
