#!/usr/bin/python

from helper_methods import *

import tensorflow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def run(files, layers, loss_function, optimizer, batch_size, epochs):
    num_gpus = len(tensorflow.config.experimental.list_physical_devices('GPU'))
    using_gpus = num_gpus >= 1
    print(f"Using GPU: {using_gpus}\n")

    output_folder = 'output/'
    input_folder = 'terrain/'
    result = ""
    for file in files:
        # Clear backend for new file
        keras.backend.clear_session()

        print(f"Working on file: {file}")
        input_path = input_folder + file
        output_path = output_folder + file + '.h5'

        # Get x and y
        x, y = get_xy(input_path)

        # Output Layer
        model = Sequential()
        # Input Layer
        model.add(Input(2))
        # Hidden Layers
        for layer in layers:
            model.add(layer)
        # Output Layer
        y_mean = y.mean()
        model.add(Lambda(lambda v: v + y_mean))

        # Callback function for early stopping
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)

        # Compile
        model.compile(optimizer=optimizer, loss=loss_function, metrics=[Entropy()])
        improvement = print_error(y, y.mean(), 1, 'Constant')
        result = "File: " + file + "\n\tLayers: " + layers + "\n\tLoss Function: " + loss_function + "\n\tImprovement: " + improvement + "\n\n"
        model.fit(x, y, batch_size=batch_size, verbose=1, epochs=epochs, callbacks=[callback])

        # Save result
        save_model(model, output_path)
        compare_images(model, x, y, output_path)
    # Opening a file
    file1 = open('results.txt', 'w+')
    file1.write(result)
    file1.close()
    print("Results written to results.txt")


def run_all(layers, loss_function, optimizer, batch_size, epochs):
    all_files = ['Appalachian_State_0.1deg.tiff',
                 'Appalachian_State_1.0deg.tiff',
                 'Appalachian_State_2.0deg.tiff',
                 'Grand_Canyon_0.1deg.tiff',
                 'Grand_Canyon_1.0deg.tiff',
                 'Grand_Canyon_2.0deg.tiff',
                 'NC_Coast_1.0deg.tiff',
                 'NC_Coast_2.0deg.tiff',
                 'NC_Coast_3.0deg.tiff']
    run(all_files, layers, loss_function, optimizer, batch_size, epochs)
