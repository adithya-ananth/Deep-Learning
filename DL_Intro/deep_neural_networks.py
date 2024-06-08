#2

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd

df = pd.read_csv(r'CSV files\concrete.csv')

# ReLU - Rectified Linear Unit is an activation function, best expressed as max(0, y)

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=512, activation='relu', input_shape=[8]),
    layers.Dense(units=512, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])