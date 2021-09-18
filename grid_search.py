#!/usr/bin/python

import itertools
from terrain_modeling import run, run_all
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

# Files to run
files = [
    # 'Appalachian_State_0.1deg.tiff',
    # 'Appalachian_State_1.0deg.tiff',
    # 'Appalachian_State_2.0deg.tiff',
    'Grand_Canyon_0.1deg.tiff',
    'Grand_Canyon_1.0deg.tiff',
    'Grand_Canyon_2.0deg.tiff',
    # 'NC_Coast_1.0deg.tiff',
    # 'NC_Coast_2.0deg.tiff',
    # 'NC_Coast_3.0deg.tiff'
]

# Hyper-parameters for Grid Search
layer_counts = [3, 4]
neuron_counts = [10, 20, 30]
loss_functions = [MeanSquaredError()]

# Constants - ACCEPTABLE ERROR
optimizer = Adamax(learning_rate=0.01)
batch_size = 1024
epochs = 200

# Run grid search
if __name__ == '__main__':
    permutations = True

    if permutations:
        for file in files:
            for loss_function in loss_functions:
                print("le")
                for layer_count in layer_counts:
                    print(layer_count)
                    perms = list(itertools.product(neuron_counts, repeat=layer_count))
                    for perm in perms:
                        layers = []
                        for neuron_amt in perm:
                            layers.append(Dense(neuron_amt, activation='relu'))
                        run([file], layers, loss_function,
                            optimizer, batch_size, epochs, False)

    else:
        for file in files:
            for loss_function in loss_functions:
                for layer_count in layer_counts:
                    for neuron_count in neuron_counts:
                        # Create layers
                        layers = []
                        for i in range(layer_count):
                            layers.append(Dense(neuron_count, activation='relu'))
                        run([file], layers, loss_function,
                            optimizer, batch_size, epochs, False)
