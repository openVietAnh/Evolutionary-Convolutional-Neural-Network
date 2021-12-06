# Top 10 models:
## ZIP file containing top 10 fully-trained models:
https://drive.google.com/file/d/1047lqcn6PvH43mScO-CSqXvrR9O3ksJd/view?usp=sharing

## Top 10 models (individual) information:
### Gene and fitness (accuracy):
| Gene                                                                  | Accuracy (Fitness) | Adjusted Fitness 
| ----------------------------------------------------------------------| -------------------| -------------------
| 001000100000011101000000111011101001010000011111001110000101000100100 | 0.9797999858856201 | 0.8164999882380168 
| 001000100000011101000001111011101001011100011111001110000101000101100 | 0.9735999703407288 | 0.8163006554047266
| 110111111100001101000000011001111001010000110111001111001111011101100 | 0.9722999930381775 | 0.9590714217043247
| 001000100000011101000000111111000001110000110111001111010111010100100 | 0.9699000120162964 | 0.8917662838732006
| 010011110111011101000000011001011001010000110111001110001111001101100 | 0.9688000082969666 | 0.9040935142363157
| 001001101000001000000000111011001001110000011111000000000011001101100 | 0.9678999781608582 | 0.8518507460854491
| 001000100000011101000000111111001001110010110111001111010111010100100 | 0.9664000272750854 | 0.8885482527899818
| 001100100100011010010010111001000001010000110111001110001110001101100 | 0.9657999873161316 | 0.8621071759006463
| 000010111110101100000000101001001011110000010111001100001111100101100 | 0.9645000100135803 | 0.8893441650774572
| 001001101000001000000000111010101001110000110111001111010101010100100 | 0.9642000198364258 | 0.8875611353381493

## Top 10 models structure:
| b  | n_c | c_k           | c_s        | c_p        | c_a                        | n_d | d_t                     | d_n     | d_a            | d_d    | d_r      | n     | f       |
| -- | --- | -----------   | ---------- | ---------- | -------------------------- | -   | ----------------------- | ------- | -------------- | ------ | -------- | ----- | ------- |
| 25 | 3   | 4, 128, 256   | 2, 6, 5    | 1, 1, 6    | linear, relu, relu         | 2   | feed-forward, recurrent | 128, 32 | linear, linear | 0, 0   | None, 11 | 0.001 | AdaMax  |
| 25 | 3   | 4, 128, 256   | 2, 6, 5    | 1, 1, 6    | linear, relu, relu         | 2   | feed-forward, recurrent | 128, 32 | linear, linear | 0, 0   | None, l1 | 0.001 | Adam    |
| 15 | 2   | 256, 128      | 9, 6       | 1, 1       | relu, relu                 | 1   | feed-forward            | 128     | linear, linear | 0.5    | None     | 0.001 | Adam    |
| 25 | 3   | 4, 128, 256   | 2, 6, 9    | 1, 1, 1    | linear, relu, relu         | 1   | feed-forward            | 128     | linear         | 0.5    | None     | 0.001 | AdaMax  |
| 50 | 1   | 256           | 7          | 7          | linear                     | 1   | feed-forward            | 128     | linear         | 0      | None     | 0.001 | Adam    |
| 25 | 3   | 16, 32, 256   | 4, 2, 5    | 1, 1, 2    | relu, relu, relu           | 2   | feed-forward, recurrent | 128, 16 | relu, linear   | 0, 0.5 | l1, 11   | 0.001 | Adam    |
| 25 | 3   | 4, 128, 256   | 2, 6, 9    | 1, 1, 2    | linear, relu, relu         | 1   | feed-forward            | 128     | linear         | 0.5    | None     | 0.001 | AdaMax  |
| 25 | 4   | 4, 64, 256, 8 | 3, 3, 3, 6 | 1, 2, 1, 2 | linear, relu, relu, linear | 1   | feed-forward            | 128     | linear         | 0      | None     | 0.001 | Adam    |
| 25 | 1   | 64            | 9          | 6          | relu                       | 1   | feed-forward            | 128     | linear         | 0      | l1 + l2  | 0.001 | Adam    |
| 25 | 3   | 16, 32, 256   | 4, 2, 4    | 1, 1, 6    | relu, relu, relu           | 1   | feed-forward            | 128     | linear         | 0.5    | None     | 0.001 | AdaGrad |

# Implementation:
## File description
EC process is ran in `evolutionary_computing.py` file.

The file used CNN class in `convolutional_neural_network.py` file to evaluate individuals (models).

`data_analysis.py` display charts to show insights from the result. Folder `resultImages` contains all the charts.

`populationHistory.txt` show the evolve process throughout 87 generations. 

`train_model.ipynb` used to fully train top 10 models.

`test_top_10.ipynb` used to evaluate all fully trained top 10 models and also ensemble model.

## Genetic Algorithm Information:
Stop condition: 100 generation limit or 30 generations without improvement

Number of individuals in the population: 50

Mutation rate in individual's gene: 0.015

Tournament size for parent selection: 3

Elite set size: 1

Number of points in multipoints crossover: from 3 to 10

Population replacement: Keep the new offspring generation, delete worst individual then add the elite individual.
