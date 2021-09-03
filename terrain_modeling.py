# Pseudo code for the program operation:
from helper_methods import *
import sys

def run():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Lambda

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
        print(f"Working on file: {file}")
        input_path = input_folder + file
        output_path = output_folder + file + '.h5'

        x, y = get_xy(input_path)

        keras.backend.clear_session()

        # linear regression
        model = Sequential()
        model.add(Input(2))  # 2 inputs: (x, y)
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))  # 1 output: height (estimated)
        # Initially the network outputs values centered at zero
        # Add the mean elevation to start near the solution
        y_mean = y.mean()
        model.add(Lambda(lambda v: v + y_mean))

        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[Entropy()])
        model.summary()

        print_error(y, y.mean(), 1, 'Constant')

        model.fit(x, y, batch_size=128, verbose=1, epochs=10)
        save_model(model, output_path)
        compare_images(model, x, y, output_path)


if __name__ == '__main__':
    run()
