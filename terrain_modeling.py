# Pseudo code for the program operation:
from helper_methods import *

def run():
    """
    An minimal example training a linear regression model on the elevation data.

    :return: None
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Lambda

    output_path = 'terrain/linear'
    tiff_file = f'terrain/Appalachian_State_0.1deg.tiff'

    model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
    x, y = get_xy(tiff_file)

    keras.backend.clear_session()

    # linear regression
    model = Sequential()
    model.add(Input(2))
    model.add(Dense(1, activation='linear'))
    # Initially the network outputs values centered at zero
    # Add the mean elevation to start near the solution
    y_mean = y.mean()
    model.add(Lambda(lambda v: v + y_mean))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[Entropy()])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    model.fit(x, y, batch_size=64, verbose=1, callbacks=[model_checkpoint])
    compare_images(model, x, y, output_path)


if __name__ == '__main__':
    run()
