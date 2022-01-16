with open("result1.txt", "r") as f:
    data = f.readlines()
ind = data.index("Generation 9\n")
population = []
for j in range(1, 51):
    items = data[ind + j].strip().split()
    population.append([items[0], float(items[1])])
population.sort(key = lambda x: x[1], reverse = True)
population = population[:10]
for item in population:
    print(item)