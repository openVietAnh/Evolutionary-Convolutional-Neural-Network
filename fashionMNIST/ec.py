from random import randint, random, sample
from cnn import CNN

POPULATION_SIZE = 50
TOURNAMENT_SIZE = 3
MIN_POINTS = 3 # Minium number of points in multipoints crossover
MAX_POINTS = 10 # Maximum number of points in multipoints crossover
MUTATION_RATE = 0.015
ELITE_SIZE = 1
GENE_LENGTH = 67

# Dictionaries to convert genotype to phenotype
LEARNING_RATE_DICT = {
    0: 1 * 10 ** (-5), 1: 5 * 10 ** (-5),
    2: 1 * 10 ** (-4), 3: 5 * 10 ** (-4),
    4: 1 * 10 ** (-3), 5: 5 * 10 ** (-3),
    6: 1 * 10 ** (-2), 7: 5 * 10 ** (-2),
}

DENSE_TYPE_DICT = {
    0: "recurrent", 1: "feed-forward"
}

REGULARIZATION_DICT = {
    0: "l1", 1: "l2", 2: "l1l2", 3: None
}

ACTIVATION_DICT = {
    0: "relu",
    1: "linear"
}

def fitness_tournament_selection(population):
    index, selected, max_fitness = 0, None, 0
    while selected is None:
        indexes = sample([i for i in range(POPULATION_SIZE)], TOURNAMENT_SIZE)
        for i in range(TOURNAMENT_SIZE):
            contestant = population[indexes[i]]
            if contestant.adjusted_fitness > max_fitness:
                selected = contestant
                index = indexes[i]
                max_fitness = contestant.adjusted_fitness
    return index, selected

def similarity_tournament_selection(population, ind, index):
    index, selected, min_similarity = 0, None, 2
    while selected is None:
        a = [i for i in range(POPULATION_SIZE)]
        a.remove(index)
        indexes = sample(a, TOURNAMENT_SIZE)
        for i in range(TOURNAMENT_SIZE):
            contestant = population[indexes[i]]
            simi = get_similarity(ind, contestant)
            if simi < min_similarity:
                selected = contestant
                index = indexes[i]
                min_similarity = simi
    return index, selected

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
        elif len(args) == 1:
            self.gene = args[0]
        else:
            self.gene = args[0]
            self.fitness = args[1]
            self.adjusted_fitness = args[2]

    def evaluate(self):
        components = self.get_components()
        self.fitness = cnn.evaluate(components)

    def mutate(self):
        for i in range(GENE_LENGTH):
            odd = random() <= 0.015
            self.gene[i] = int(not self.gene[i]) if odd else self.gene[i]

    def get_batch_size(self):
        return [25, 50, 100, 15][binary_to_decimal(self.gene[:2])]

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
            binary = self.gene[45 + i * 8: 45 + i * 8 + 1]
            result.append(DENSE_TYPE_DICT[binary_to_decimal(binary)])
        return result

    def get_neurons_num(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[46 + i * 8: 46 + i * 8 + 3]
            result.append(2 ** (binary_to_decimal(binary) + 3))
        return result

    def get_dense_activation(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[49 + i * 8: 49 + i * 8 + 1]
            result.append(ACTIVATION_DICT[binary_to_decimal(binary)])
        return result

    def get_regularization(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[50 + i * 8: 50 + i * 8 + 2]
            result.append(binary_to_decimal(binary))
        return result

    def get_dropout(self, layers_num):
        result = []
        for i in range(layers_num):
            binary = self.gene[52 + i * 8: 52 + i * 8 + 1]
            result.append(binary_to_decimal(binary) / 2)
        return result

    def get_optimizer(self):
        binary = self.gene[61: 64]
        return binary_to_decimal(binary)

    def get_learning_rate(self):
        binary = self.gene[64: 67]
        return LEARNING_RATE_DICT[binary_to_decimal(binary)]

    def get_components(self):
        dct = {}

        dct["b"] = self.get_batch_size()

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
        dct["dd"] = self.get_dropout(dct["nd"])
        dct["dr"] = self.get_regularization(dct["nd"])

        # Learning parameters
        dct["n"] = self.get_learning_rate()
        dct["f"] = self.get_optimizer()

        return dct

    def get_str_model(self):
        b = self.get_batch_size()

        # Convolutional layers
        nc = self.get_convol_layers_num()
        ck = self.get_kernels_num(nc)
        cs = self.get_kernel_sizes(nc)
        cp = self.get_pooling(nc)
        ca = self.get_convol_activation(nc)

        # Dense layers
        nd = self.get_dense_layers_num()
        dt = self.get_dense_type(nd)
        dn = self.get_neurons_num(nd)
        da = self.get_dense_activation(nd)
        dd = self.get_dropout(nd)
        dr = self.get_regularization(nd)

        # Learning parameters
        n = self.get_learning_rate()
        f = self.get_optimizer()

        result = str(b) + "-" + str(nc) + "-"
        result += "-".join(list(map(str, ck))) + "-"
        result += "-".join(list(map(str, cs))) + "-"
        result += "-".join(list(map(str, cp))) + "-"
        result += "-".join(list(map(str, ca))) + "-"
        result += str(nd) + "-"
        result += "-".join(list(map(str, dt))) + "-"
        result += "-".join(list(map(str, dn))) + "-"
        result += "-".join(list(map(str, da))) + "-"
        result += "-".join(list(map(str, dd))) + "-"
        result += "-".join(list(map(str, dr))) + "-"
        result += str(n) + "-" + str(f)

        return result 

    def to_string(self):
        return "".join(map(str, self.gene))


class Tracker(object):
    def __init__(self, *args):
        self.generation_count = 1
        self.best_fitness = []
        self.population_history = []
        self.best_individual = None
        self.similarity_matrix = []
        self.duplicated_set = {}

    def elitism(self):
        return self.best_individual

    def update_elitism(self, population):
        generation = []
        for individual in population:
            generation.append(" ".join(["".join(map(str, individual.gene)), str(individual.fitness), str(individual.adjusted_fitness)]))
        self.population_history.append(generation)
        best, _max = None, 0
        for individual in population:
            if individual.fitness > _max:
                best, _max = individual, individual.fitness
        self.best_individual = best
        self.best_fitness.append(_max)

    def print(self):
        print(self.best_fitness)
        print(self.best_individual.gene, self.best_individual.fitness, self.best_individual.adjusted_fitness)
        for index, gen in enumerate(self.population_history):
            print("Generation", index)
            for ind in gen:
                print(ind)
            for line in self.similarity_matrix[index]:
                print(" ".join(list(map(str, line))))
        for key, value in self.duplicated_set.items():
            print(key, value)

class Population(object):
    def __init__(self, *args):
        if not args: # New initialization from start
            self.populace = [Individual() for i in range(POPULATION_SIZE)]
        else: # Create a population from populace read from a text file
            self.populace = args[0]  

    def calculate_ajusted_fitness(self, tracker, update_matrix):
        similarity_mat = [[1 for i in range(len(self.populace))] for j in range(len(self.populace))]
        for i in range(len(self.populace)):
            similarity_sum = 0
            for j in range(len(self.populace)):
                if j != i:
                    sim = get_similarity(self.populace[i], self.populace[j])
                    similarity_mat[i][j] = similarity_mat[j][i] = sim
                    similarity_sum += sim
            similarity = 1 - (similarity_sum / (len(self.populace) - 1))
            self.populace[i].adjusted_fitness = self.populace[i].fitness
            self.populace[i].adjusted_fitness *= similarity
        if update_matrix:
            tracker.similarity_matrix.append(similarity_mat)

    def print(self):
        for individual in self.populace:
            print(individual.to_string(), end = " ")
            print(individual.fitness)

def get_individual_info_from_string(line):
    items = line.strip().split()
    gene = list(map(int, list(items[0])))
    fitness = float(items[1])
    adjusted_fitness = float(items[2])
    return gene, fitness, adjusted_fitness

cnn = CNN()

population = Population()
tracker = Tracker()

# Remove all invalid individual (invalid CNN model structure)
for i in range(POPULATION_SIZE):
    population.populace[i].evaluate()
    while population.populace[i].fitness == 0:
        population.populace[i] = Individual()
        population.populace[i].evaluate()
population.calculate_ajusted_fitness(tracker, True)

tracker.update_elitism(population.populace)

print("Generation 0 done")

# Population evolution
for i in range(1, 10):
    print("".join(list(map(str, tracker.best_individual.gene))), tracker.best_individual.fitness)

    # Create parent pool for mating by tournament selection
    pool = []
    for j in range(POPULATION_SIZE):
        ind, firstParent = fitness_tournament_selection(population.populace)
        pool.append(firstParent)
        ind2, secondParent = similarity_tournament_selection(population.populace, firstParent, ind)
        pool.append(secondParent)

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

    population.populace += next_generation
    population.populace.sort(key = lambda x: x.fitness, reverse=True)
    for k in range(POPULATION_SIZE - 1):
        count, j = 0, k + 1
        while j < len(population.populace):
            while j < len(population.populace) and len(population.populace) > 50 and get_similarity(population.populace[k], population.populace[j]) == 1:
                del population.populace[j]
                count += 1
            j += 1
        if count > 0:
            model = next_generation[k].get_str_model()
            if model in tracker.duplicated_set.keys():
                tracker.duplicated_set[model] += count + 1
            else:
                tracker.duplicated_set[model] = count + 1
    
    while len(population.populace) > 50:
        population.calculate_ajusted_fitness(tracker, False)
        population.populace.sort(key = lambda x: x.adjusted_fitness, reverse=True)
        population.populace.pop()
    # population.populace = population.populace[:50]
    population.calculate_ajusted_fitness(tracker, True)
    tracker.update_elitism(population.populace)
    tracker.generation_count += 1
    print("Generation", i, "done")

tracker.print()
