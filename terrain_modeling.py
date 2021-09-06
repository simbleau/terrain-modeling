#!/usr/bin/python

from helper_methods import *
import sys

from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def run():
    output_folder = 'output/'
    input_folder = 'terrain/'
    tiff_files = ['Appalachian_State_0.1deg.tiff',
                  'Appalachian_State_1.0deg.tiff',
                  'Appalachian_State_2.0deg.tiff',
                  'Grand_Canyon_0.1deg.tiff',
                  'Grand_Canyon_1.0deg.tiff',
                  'Grand_Canyon_2.0deg.tiff',
                  'NC_Coast_1.0deg.tiff',
                  'NC_Coast_2.0deg.tiff',
                  'NC_Coast_3.0deg.tiff']

    # Only render 1 file if asked to
    if len(sys.argv) == 2:
        tiff_files = [sys.argv[1]]

    for file in tiff_files:
        # Clear backend for new file
        keras.backend.clear_session()

        print(f"Working on file: {file}")
        input_path = input_folder + file
        output_path = output_folder + file + '.h5'

        # Get x and y
        x, y = get_xy(input_path)

        # Linear Regression
        model = Sequential()
        # Input
        model.add(Input(2))  # 2 inputs: (x, y)
        # Layers
        model.add(Dense(50, activation='tanh'))
        model.add(Dense(40, activation='tanh'))
        # Output Layer
        model.add(Dense(1, activation='linear'))  # 1 output: height (estimated)
        # Initially the network outputs values centered at zero
        # Add the mean elevation to start near the solution
        y_mean = y.mean()
        model.add(Lambda(lambda v: v + y_mean))

        # Loss functions
        mse = MeanSquaredError()
        mae = MeanAbsoluteError()

        # Optimizers
        sgd = SGD(clipvalue=1)
        adamx = Adamax(learning_rate=0.0009)

        # Compile
        model.compile(optimizer=adamx, loss=mae, metrics=[Entropy()])
        model.summary()

        print_error(y, y.mean(), 1, 'Constant')

        model.fit(x, y, batch_size=128, verbose=1, epochs=20)

        save_model(model, output_path)
        compare_images(model, x, y, output_path)


if __name__ == '__main__':
    run()
