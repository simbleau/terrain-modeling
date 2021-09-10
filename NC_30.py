#!/usr/bin/python

import sys
from terrain_modeling import run, run_all
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

# File
files = ['NC_Coast_3.0deg.tiff']

# Model
layers = [
    Dense(30, activation='softmax'),
    Dense(10, activation='relu'),
    Dense(5, activation='relu'),
    Dense(1, activation='linear')
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adamax(learning_rate=0.01)

# Batch Size
batch_size = 1024

# Epochs
epochs = 25

# Run
if __name__ == '__main__':
    # Use -a flag in CLI args to run this model on all files.
    if len(sys.argv) == 2 and sys.argv[1] == "-a":
        run_all(layers, loss_function, optimizer, batch_size, epochs)
    else:
        run(files, layers, loss_function, optimizer, batch_size, epochs)