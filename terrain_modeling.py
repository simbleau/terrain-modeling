# Pseudo code for the program operation:


# Get file path for image TIFF
import sys
args = sys.argv
if len(args) != 2:
    print("Enter a file name.")
    sys.exit(1)
data_path = args[1]

# Create and train a new model instance.
model = create_model(data_path)
model.fit(...)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
# More here: https://www.tensorflow.org/tutorials/keras/save_and_load
model.save(data_path + '.h5')
