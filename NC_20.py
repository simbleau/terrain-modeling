#!/usr/bin/python

import sys
from terrain_modeling import run, run_all
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

# File
files = ['NC_Coast_2.0deg.tiff']

# Model
layers = [
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
]

# Loss function
loss_function = MeanAbsoluteError()

# Optimizers
optimizer = Adamax(learning_rate=0.01)

# Batch Size
batch_size = 512

# Epochs
epochs = 1000


# Run
if __name__ == '__main__':
    # Use -a flag in CLI args to run this model on all files.
    if len(sys.argv) == 2 and sys.argv[1] == "-a":
        run_all(layers, loss_function, optimizer, batch_size, epochs)
    else:
        run(files, layers, loss_function, optimizer, batch_size, epochs)
