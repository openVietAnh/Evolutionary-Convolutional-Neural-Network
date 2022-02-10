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
#     '110110110001111100010001111011011111100110010110110100111110101011111', # 0 - 1
#     '110010111001100110010001111011011110001010010110110100111110101011111', # 2 - 2
#     '110110110001111100010001111011011111100011110110110000001110101011111', # 3 - 3
#     '110010110101100010011111111001011100101100011110110100111101101011111', # 4 - 4
#     '110010110001100011011111100100111111101000011110110000111110101011111', # 5 - 5
#     '110010111001100110010111110100110110001000011110100000111110101011111', # 6 - 6
#     '110010110101100110011111100111011111100100110110110000111100000011111', # 9 - 7
#     '110010111001100111110111110100110110100011110110111000001100000100110', # 11 - 8
#     '110010111001100111010111110100110110100011110110111000001100101011111', # 12 - 9
#     '110010110001100011011110110011011100101100011110110100111110101011111', # 13 - 10
# ]

GENE = [
    ['110010010101001011000101001110110001001100100111100010110110111100101', 0.9807000160217285, 0.8729716051708568],
    ['111101000100001000000001110101000110101110100111110010110111001100100', 0.9804999828338623, 0.9058967232704163],
    ['100111010000101011000101001100011100110101100111100011111101101100101', 0.9804999828338623, 0.8225305411550734],
    ['100111010000101011000101000100011100110101100111001011111100011100101', 0.9793999791145325, 0.8107255382670298],
    ['111101000100001000000001001001000110101110100111110010110111001100101', 0.9793999791145325, 0.9084289661352185],
    ['110110010000101011000101000000011100110101000111001111111100011100101', 0.9790999889373779, 0.8186363796393077],
    ['110110010000101011100101001000010100110100101111001101111100001100101', 0.9790999889373779, 0.846513532102108],
    ['110110010000101011000101000100100001101101010111001111111111010010111', 0.9789000153541565, 0.8293458463417159],
    ['110010010001001011100101001100110000110101101111100010110100110100101', 0.9783999919891357, 0.8255249932408333],
    ['110010010001101011100101001000010100110100101111100011111100111100101', 0.9783999919891357, 0.8280729098866383],
    ['110010010001101011100101001010110000110101101111100010111100100100101', 0.9783999919891357, 0.8306208265324434],
    ['100111010000101011000101001100011100110101101111100011111101101100101', 0.9782000184059143, 0.8763041831552982],
    ['110010010101001011001101001110110000110101101111100011111100111100101', 0.9778000116348267, 0.8377505308017135]
]

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

s = set()
a = []
for index, gene in enumerate(GENE):
    des = ""
    dct = get_components(gene[0])
    for item in dct.values():
        des += str(item) + "-"
    if des not in s:
        print(gene, index)
        a.append(des)
        s.add(des)
print(len(a), len(s))
for item in a:
    print(item)