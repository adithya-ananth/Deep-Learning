#1

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd 

df = pd.read_csv(r'CSV files\red-wine.csv')

# Get rows and columns
size = df.shape

# The input shape is a list containing the number of features (here, the number of columns - 1)
input_shape = [11]

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])

# Look at the weights and bias
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))