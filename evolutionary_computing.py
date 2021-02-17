from random import randint, random

POPULATION_SIZE = 50
MAXIMUM_GENERATION = 100
STOP_CONDITION = 30 # Number of generations without improvements
TOURNAMENT_SIZE = 3
MIN_POINTS = 3 # Minium number of points in multipoints crossover
MAX_POINTS = 10 # Maximum number of points in multipoints crossover
MUTATION_RATE = 0.015
ELITE_SIZE = 1
GENE_LENGTH = 69

def tournament_selection(population):
    selected, max_fitness = None, 0
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

def calculate_ajusted_fitness(fitness):
    pass


class Individual(object):
    def __init__(self, *args):
        if not args: # New initialization from start
            self.gene = [randint(0, 1) for i in range(GENE_LENGTH)]
        else:
            self.gene = args[0]

    def evaluate(self):
        components = self.get_components()
        self.fitness = randint(0, 100)
        self.adjusted_fitness = calculate_ajusted_fitness(self.fitness)

    def mutate(self):
        for i in range(GENE_LENGTH):
            odd = random() <= 0.015
            self.gene[i] = int(not self.gene[i]) if odd else self.gene[i]

    def get_convol_layers_num(self):
        pass

    def get_kernels_num(self):
        pass

    def get_kernel_sizes(self):
        pass

    def get_pooling(self):
        pass

    def get_convol_activation(self):
        pass

    def get_dense_layers_num(self):
        pass

    def get_dense_type(self):
        pass

    def get_neurons_num(self):
        pass

    def get_dense_activation(self):
        pass

    def get_dropout(self):
        pass

    def get_regularization(self):
        pass

    def get_optimizer(self):
        pass

    def get_learning_rate(self):
        pass

    def get_components(self):
        dct = {}

        # Convolutional layers
        dct["nc"] = self.get_convol_layers_num()
        dct["ck"] = self.get_kernels_num()
        dct["cs"] = self.get_kernel_sizes()
        dct["cp"] = self.get_pooling()
        dct["ca"] = self.get_convol_activation()

        # Dense layers
        dct["nd"] = self.get_dense_layers_num()
        dct["dt"] = self.get_dense_type()
        dct["dn"] = self.get_neurons_num()
        dct["da"] = self.get_dense_activation()
        dct["dr"] = self.get_dropout()
        dct["dd"] = self.get_regularization()

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
        if self.generation_count >= 30:
            if self.best_fitness[-1] == self.best_fitness[-30]:
                return True
        return False

    def elitism(self):
        return self.best_individual

    def update_elitism(self, population):
        best, _max = None, 0
        for individual in population:
            if individual.fitness > _max:
                best, _max = individual, individual.fitness
        self.best_individual = best
        self.best_fitness.append(_max)


class Population(object):
    def __init__(self, *args):
        if not args: # New initialization from start
            self.populace = [Individual() for i in range(POPULATION_SIZE)]
        else: # Create a population from text file
            file_name = args[0]

    def print(self):
        for individual in self.populace:
            print(individual.to_string())


population = Population()
# population.print()

# Remove all invalid individual (invalid CNN model structure)
for i in range(POPULATION_SIZE):
    population.populace[i].evaluate()
    while population.populace[i].fitness == 0:
        population.populace[i] = Individual()
        population.populace[i].evaluate()

tracker = Tracker()
tracker.update_elitism(population.populace)

# Population evolution
for i in range(MAXIMUM_GENERATION):
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
    tracker.update_elitism(population.populace)
    tracker.generation_count += 1
    population.print()

    # Got 30 generations without improvements
    if tracker.stop_condition():
        break
