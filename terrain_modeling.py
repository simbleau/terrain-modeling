#!/usr/bin/python

from helper_methods import *

import tensorflow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def run(files, layers, loss_function, optimizer, batch_size, epochs, save):
    # Printing Debug information
    num_gpus = len(tensorflow.config.experimental.list_physical_devices('GPU'))
    using_gpus = num_gpus >= 1
    print(f"Using GPU: {using_gpus}\n")

    # Debug variables
    layers_str = "[" + ",".join(str(x.units) for x in layers) + "]"
    loss_function_name = loss_function.name

    # Begin run
    output_folder = 'output/'
    input_folder = 'terrain/'

    for file in files:
        # Clear backend for new file
        keras.backend.clear_session()

        input_path = input_folder + file
        output_path = output_folder + file + '.h5'

        # Print debug info before starting
        print(f"Working on file: {file}")
        if save:
            print(f"Output: {output_path}")
        print(f"Hyper-parameters:\n\t" +
              f"Layers: {layers_str}\n\tLoss Function: {loss_function_name}\n\tBatch Size: {batch_size}\n\t" +
              f"Epochs: {epochs}")

        # Get x and y
        x, y = get_xy(input_path)

        # Output Layer
        model = Sequential()
        # Input Layer
        model.add(Input(2))
        # Hidden Layers
        for layer in layers:
            model.add(layer)

        model.add(Dense(1, activation='relu', name='output'))
        # Output Layer
        y_mean = y.mean()
        model.add(Lambda(lambda v: v + y_mean))

        # Callback function for early stopping
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=15)

        # Compile
        model.compile(optimizer=optimizer,
                      loss=loss_function, metrics=[Entropy()])
        print_error(y, y.mean(), 1, 'Constant')

        model.fit(x, y, batch_size=batch_size, verbose=1,
                  epochs=epochs, callbacks=[callback])

        # Save result
        if save:
            model.save(output_path)
            compare_images(model, x, y, output_path)

        # Write results and CSV
        improvement = get_improvement(model, x, y)

        result = f"File: {file}\n\tLayers: {layers_str}\n\tLoss Function: {loss_function_name}\n\tBatch Size: {batch_size}\n\tEpochs: {epochs}\n\tImprovement: {improvement}\n"
        file1 = open('results.txt', 'a+')
        file1.write(result)
        file1.close()

        csv_result = f"{file},{layers_str},{loss_function_name},{batch_size},{epochs},{improvement}\n"
        file1 = open('results.cvs', 'a+')
        file1.write(csv_result)
        file1.close()

        print("Results appended.\n")


def run_all(layers, loss_function, optimizer, batch_size, epochs, save):
    all_files = ['Appalachian_State_0.1deg.tiff',
                 'Appalachian_State_1.0deg.tiff',
                 'Appalachian_State_2.0deg.tiff',
                 'Grand_Canyon_0.1deg.tiff',
                 'Grand_Canyon_1.0deg.tiff',
                 'Grand_Canyon_2.0deg.tiff',
                 'NC_Coast_1.0deg.tiff',
                 'NC_Coast_2.0deg.tiff',
                 'NC_Coast_3.0deg.tiff']
    run(all_files, layers, loss_function, optimizer, batch_size, epochs, save)
