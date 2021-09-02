import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.utils import layer_utils


def error_bits(error):
    """
    Return a lower bound on the number of bits to encode the errors based on Shannon's source
    coding theorem:
    https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem#Source_coding_theorem

    :param error: Vector or list of errors (error = estimate - actual)
    :return: The lower bound number of bits to encode the errors
    """
    # round and cast to an integer, reshape as a vector
    error = np.round(error).astype(int).reshape((-1,))

    # shift so that the minimum value is zero
    error = error - error.min()

    # count how many occurrences of each discrete value
    p = np.bincount(error)

    # ignore zero counts
    p = p[p > 0]

    # convert counts into discrete probability distribution
    p = p / p.sum()

    # compute entropy (bits per codeword):
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    entropy = -(p * np.log2(p)).sum()

    # minimum bits to encode all errors
    # https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem#Source_coding_theorem
    bits = int(np.ceil(entropy * len(error)))

    return bits


def get_xy(tiff_file):
    import matplotlib.pyplot as plt
    """
    Read TIFF file and convert to data matrix, X and target vector, y.

    :param tiff_file: Image file name (string)
    :return: data matrix 'X' (129600-by-2), target vector 'y' (129600-by-1)
    """

    # Read the image
    image = plt.imread(tiff_file)

    # get normalized coordinate system:
    # (-180, -180) in upper-left (northwest) corner,
    # (179, 179) in lower-right (southeast) corner.
    coordinates = range(-180, 180)
    i, j = np.meshgrid(coordinates, coordinates, indexing='ij')

    # reshape row/column indexes into n-by-2 matrix
    x = np.concatenate((i.reshape((-1, 1)), j.reshape((-1, 1))), axis=1).astype('float32')

    # reshape elevations into n-by-1 vector
    y = image.reshape((-1, 1)).astype('float32')

    return x, y


# bogus
def print_error(y_true, y_pred, num_params, name):
    """
    Print performance for this model.

    :param y_true: The correct elevations (in meters)
    :param y_pred: The estimated elevations by the model (in meters)
    :param num_params: The number of trainable parameters in the model
    :param name: The name of the model
    :return: None
    """
    # the error
    e = y_pred - y_true
    # the mean squared error
    loss = (e ** 2).mean()
    # the lowerbound for the number of bits to encode the errors
    bits = error_bits(e)
    # the number of bits to encode the model
    desc = 32 * num_params
    # the total bits for the compressed image
    total = desc + bits
    # comparison to a model that estimated zero for every pixel
    constant_bits = error_bits(y_true)
    # percent improvement
    improvement = 1 - total/constant_bits
    print(f'{name:8s} MSE: {loss:11.4f}, error bits/px: {bits/len(e):11.4f}, model bits/px: {desc/len(e):11.4f}, '
          f'total bits/px: {total/len(e):11.4f}, improvement = {improvement:.2%}')


# bogus
def model_bits(model):
    """
    Compute the number of bits to encode the model based on the trainable parameters.
    (We ignore the bits required to encode the architecture for the model.)

    :param model: A keras model
    :return: The number of bits required to encode the model.
    """
    return 32 * layer_utils.count_params(model.trainable_weights)


def compare_images(model, x, y, output_path):
    """
    Use the model to estimate elevation image and show it compared
    to the original.

    :param model: A keras model with 2D input and 1 output
    :param x: The (i, j) coordinates
    :param y: The target vector
    :return: None
    """
    # The model estimates (same shape as 'y')
    y_hat = model.predict(x)
    error = y_hat - y
    mse = np.mean(error**2)
    err_bits = error_bits(error)
    num_trainable_params = layer_utils.count_params(model.trainable_weights)
    mod_bits = 32 * num_trainable_params
    orig_bits = error_bits(y)

    # print some comparisons
    print_error(y, 0, 0, 'Zero')
    print_error(y, y.mean(), 1, 'Constant')
    print_error(y, LinearRegression().fit(x, y).predict(x), 3, 'Linear')
    print_error(y, y_hat, layer_utils.count_params(model.trainable_weights), 'Model')

    improvement = 1 - (mod_bits + err_bits)/orig_bits

    # The minimum and maximum elevation over original and estimated image
    vmin = min(y.min(), y_hat.min())
    vmax = max(y.max(), y_hat.max())

    # The (i, j) extents (-180 to 179)
    extent = (-180.5, 179.5, 179.5, -180.5)

    # close all figure windows
    plt.close('all')

    # create a new figure window with room for 2 adjacent images and a color bar
    fig, ax = plt.subplots(1, 2, sharey='row', figsize=(10, 4))

    # render and label the model estimate
    ax[0].imshow(y_hat.reshape((360, 360)), vmin=vmin, vmax=vmax, extent=extent)
    ax[0].set_title(f'Model Estimate MSE = {mse:.1f}\n'
                    f'entropy = {err_bits/len(y):.4f}, model_bits/px = {mod_bits/len(y):.4f}\n'
                    f'total bits/px = {(err_bits + mod_bits)/len(y):.4f}, '
                    f'improvement = {improvement:.2%}')

    # render and label the original
    im1 = ax[1].imshow(y.reshape((360, 360)), vmin=vmin, vmax=vmax, extent=extent)
    ax[1].set_title('Original')
    ax[1].set_title(f'PNG Estimate MSE = {np.mean(y**2):.1f}\n'
                    f'entropy = {orig_bits/len(y):.4f}, model_bits/px = {0:.4f}\n'
                    f'total bits/px = {orig_bits/len(y):.4f}, '
                    f'improvement = {0:.2%}')

    # add a color bar in a new set of axes
    fig.subplots_adjust(right=0.8)
    ax2 = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im1, cax=ax2)

    plt.savefig(output_path + '.png')


class Entropy(keras.metrics.Metric):
    """
    This class provides a custom *metric* for the Shannon entropy of the error.
    It can be compiled as part of the model. During training, error bits per pixel
    get printed alongside the loss.
    """
    max_error = 2 ** 22

    def __init__(self, name="entropy", **kwargs):
        """
        Initialize the vector of zeros to keep track of the count of errors between -max_error and max_error.
        A limitation is that all errors outside these bounds get mapped to the nearest extreme, thereby
        reducing the computed entropy.

        :param name: This is the name that tensorflow uses to identify the metric. For example, "entropy" can be
                     monitored by a ModelCheckpoint.
        :param kwargs: Whatever else tensorflow might use to instantiate the ancestor classes
        """
        super(Entropy, self).__init__(name=name, **kwargs)
        # initialize a length 2*max_error+1 vector to hold the errors from -max_error to max_error
        self.error_counts = self.add_weight(
            name='error_counts',
            shape=(2*self.max_error+1,),
            initializer='zeros',
            dtype='int32'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the error counts based on this batch of results.

        Raises ValueError if sample_weights are applied.

        :param y_true: The correct elevations
        :param y_pred: The predicted elevations
        :param sample_weight: In general, a metric should handle weighted samples. We do not.
        :return:
        """
        if sample_weight is not None:
            raise NotImplementedError('Entropy metric does not support sample weights.')

        # make sure these are signed integers
        y_pred = tf.cast(tf.round(y_pred), 'int32')
        y_true = tf.cast(y_true, 'int32')

        # compute error
        error = tf.subtract(y_pred, y_true)

        # clip to range
        clipped_to_range = tf.clip_by_value(error, -self.max_error, self.max_error)

        # make these positive
        error = tf.add(self.max_error, clipped_to_range)

        # count the occurence of each error value
        error_counts = tf.math.bincount(
            error,
            minlength=2*self.max_error+1,
            maxlength=2*self.max_error+1
        )
        # update total counts
        self.error_counts.assign_add(error_counts)

    def result(self):
        """
        Compute the entropy from the error counts.
        :return: Entropy scalar
        """
        # cast to float before dividing
        error_counts = tf.cast(self.error_counts, 'float32')
        total_count = tf.reduce_sum(error_counts)
        probabilities = tf.math.divide(error_counts, total_count)
        # replace zeros with ones for entropy computation (avoid log(0))
        probabilities = tf.where(tf.equal(probabilities, 0), tf.ones_like(probabilities), probabilities)
        # compute entropy
        entropy = tf.reduce_sum(-(probabilities * tf.math.log(probabilities)/tf.math.log(2.)))
        return entropy

    def reset_states(self):
        """
        Reset the metric by setting error_counts to zero.

        :return: None
        """
        # The state of the metric will be reset at the start of each epoch.
        self.error_counts.assign(tf.zeros(self.error_counts.shape, dtype='int32'))

    # def get_config(self):
    #     return super(Entropy, self).get_config()


def setup_model_checkpoints(output_path, save_freq):
    """
    Setup model checkpoints using the save path and frequency.

    :param output_path: The directory to store the checkpoints in
    :param save_freq: The frequency with which to save them "epoch" means each epoch
                      See ModelCheckpoint documentation.
    :return: a ModelCheckpoint
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'model.{epoch:05d}_{entropy:.4f}.h5'),
        # os.path.join(output_path, 'model.{epoch:05d}.h5'),
        save_weights_only=False,
        save_freq=save_freq,
        monitor='entropy',
        verbose=1
    )
    return model_checkpoint


def save_model(model, file_name):
    """
    Save a trained model.

    :param model: The keras model
    :param file_name: the name to use to save it (ending in .h5)
    :return: None
    """
    model.save(file_name)


def load_model(file_name):
    """
    Load a saved model optionally including the Entropy custom metric.

    :param file_name: The file name containing the model (ends in .h5)
    :return: A keras model ready for fitting or predicting
    """
    keras.utils.get_custom_objects()['Entropy'] = Entropy
    model = keras.models.load_model(
        file_name,
        compile=True,
        custom_objects={}
    )
    return model


def example():
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
    example()
