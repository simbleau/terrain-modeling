#!/usr/bin/python

import sys
from terrain_modeling import run, run_all
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

# Files to run
files = [
    'Appalachian_State_0.1deg.tiff',
    'Appalachian_State_1.0deg.tiff',
    'Appalachian_State_2.0deg.tiff',
    'Grand_Canyon_0.1deg.tiff',
    'Grand_Canyon_1.0deg.tiff',
    'Grand_Canyon_2.0deg.tiff',
    'NC_Coast_1.0deg.tiff',
    'NC_Coast_2.0deg.tiff',
    'NC_Coast_3.0deg.tiff'
]

# Hyper-parameters for Grid Search
layer_counts = [3, 5, 7]
neuron_counts = [5, 10, 20]
loss_functions = [MeanSquaredError(), MeanAbsoluteError()]

# Constants - ACCEPTABLE ERROR
optimizer = Adamax(learning_rate=0.01)
batch_size = 1024
epochs = 1000

# Run grid search
if __name__ == '__main__':
    for layer_count in layer_counts:
        for neuron_count in neuron_counts:
            for loss_function in loss_functions:
                # Create layers
                layers = []
                for i in range(layer_count):
                    layers.append(Dense(neuron_count, activation='relu'))
                run(files, layers, loss_function,
                    optimizer, batch_size, epochs, False)
