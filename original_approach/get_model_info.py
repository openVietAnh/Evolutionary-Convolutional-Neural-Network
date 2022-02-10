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

ACTIVATION_DICT = {
    0: "relu",
    1: "linear"
}

# GENE = [
#     '001000100000011101000000111011101001010000011111001110000101000100100',
#     '001000100000011101000001111011101001011100011111001110000101000101100', 
#     '110111111100001101000000011001111001010000110111001111001111011101100', 
#     '001000100000011101000000111111000001110000110111001111010111010100100', 
#     '010011110111011101000000011001011001010000110111001110001111001101100', 
#     '001001101000001000000000111011001001110000011111000000000011001101100', 
#     '001000100000011101000000111111001001110010110111001111010111010100100',  
#     '001100100100011010010010111001000001010000110111001110001110001101100',
#     '000010111110101100000000101001001011110000010111001100001111100101100', 
#     '001001101000001000000000111010101001110000110111001111010101010100100'
# ]

GENE = ["011001100000011101000000111011101001010000011110101110001101011101100"]

def binary_to_decimal(bits):
    return int("".join(map(str, bits)), 2)

def get_batch_size(gene):
        return [25, 50, 100, 15][binary_to_decimal(gene[:2])]

def get_convol_layers_num(gene):
    return 1 + binary_to_decimal(gene[2:4])

def get_kernels_num(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[4 + i * 10: 4 + i * 10 + 3]
        result.append(2 ** (binary_to_decimal(binary) + 1))
    return result

def get_kernel_sizes(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[7 + i * 10: 7 + i * 10 + 3]
        result.append(2 + binary_to_decimal(binary))
    return result

def get_pooling(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[10 + i * 10: 10 + i * 10 + 3]
        result.append(1 + binary_to_decimal(binary))
    return result

def get_convol_activation(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[13 + i * 10: 13 + i * 10 + 1]
        result.append(ACTIVATION_DICT[binary_to_decimal(binary)])
    return result

def get_dense_layers_num(gene):
    return 1 + binary_to_decimal([gene[44]])

def get_dense_type(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[45 + i * 9: 45 + i * 9 + 2]
        result.append(DENSE_TYPE_DICT[binary_to_decimal(binary)])
    return result

def get_neurons_num(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[47 + i * 9: 47 + i * 9 + 3]
        result.append(2 ** (binary_to_decimal(binary) + 3))
    return result

def get_dense_activation(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[50 + i * 9: 50 + i * 9 + 1]
        result.append(ACTIVATION_DICT[binary_to_decimal(binary)])
    return result

def get_regularization(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[51 + i * 9: 51 + i * 9 + 2]
        result.append(binary_to_decimal(binary))
    return result

def get_dropout(gene, layers_num):
    result = []
    for i in range(layers_num):
        binary = gene[53 + i * 9: 53 + i * 9 + 1]
        result.append(binary_to_decimal(binary) / 2)
    return result

def get_optimizer(gene):
    binary = gene[63: 66]
    return binary_to_decimal(binary)

def get_learning_rate(gene):
    binary = gene[66: 69]
    return LEARNING_RATE_DICT[binary_to_decimal(binary)]

def get_components(gene):
    dct = {}

    dct["b"] = get_batch_size(gene)

    # Convolutional layers
    dct["nc"] = get_convol_layers_num(gene)
    dct["ck"] = get_kernels_num(gene, dct["nc"])
    dct["cs"] = get_kernel_sizes(gene, dct["nc"])
    dct["cp"] = get_pooling(gene, dct["nc"])
    dct["ca"] = get_convol_activation(gene, dct["nc"])

    # Dense layers
    dct["nd"] = get_dense_layers_num(gene)
    dct["dt"] = get_dense_type(gene, dct["nd"])
    dct["dn"] = get_neurons_num(gene, dct["nd"])
    dct["da"] = get_dense_activation(gene, dct["nd"])
    dct["dd"] = get_dropout(gene, dct["nd"])
    dct["dr"] = get_regularization(gene, dct["nd"])

    # Learning parameters
    dct["n"] = get_learning_rate(gene)
    dct["f"] = get_optimizer(gene)

    return dct

for gene in GENE:
    dct = get_components(gene)
    for key, value in dct.items():
        print(key, value)
    print()