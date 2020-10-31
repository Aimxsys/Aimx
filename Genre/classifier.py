import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "dataset_c1_f1.json"

def load_data(dataset_path):
    """
    Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    with open(dataset_path, "r") as file:
        data = json.load(file)

    # convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    labels = np.array(data["labels"])

    print("Data loaded succesfully.")

    return  inputs, labels

if __name__ == "__main__":

    inputs, labels = load_data(DATA_PATH)

    # create train/test split
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size = 0.3)

    # build network topology
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation = 'relu'),
        keras.layers.Dense(256, activation = 'relu'),
        keras.layers.Dense( 64, activation = 'relu'),
        keras.layers.Dense( 10, activation = 'softmax')
    ])

    # compile model
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
                  loss      = 'sparse_categorical_crossentropy',
                  metrics   = ['accuracy'])

    model.summary()

    # train model
    history = model.fit(inputs_train, labels_train, validation_data = (inputs_test, labels_test), batch_size = 32, epochs = 50)