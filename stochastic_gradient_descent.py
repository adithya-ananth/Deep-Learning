#3

'''
Each example in the training data consists of some features (the inputs) together with an expected 
target (the output). Training the network means adjusting its weights in such a way that it can 
transform the features into the target.

In addition to the training data, we need two more things:
1. A "Loss Function" that measures how good the network's predictions are.
2. An "Optimizer" that can tell the network how to change its weights.
'''

'''
During training, the model will use the loss function as a guide for finding the correct values of 
its weights (lower loss is better). In other words, the loss function tells the network its objective.

The Optimizer is an algorithm that adjusts the weights to minimize the loss.
'''

'''
Each iteration's sample of training data is called a Minibatch (or often just "Batch"), while a 
complete round of the training data is called an Epoch.

The regression line makes small shifts in the direction of each batch, and the size of each shift 
is determined by the 'Learning Rate'.
A smaller learning rate means the network needs to see more minibatches before its weights converge 
to their best values.

The learning rate and the size of the minibatches are the two parameters that have the largest 
effect on how the SGD training proceeds.
'''

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'CSV files\fuel.csv')
X = df.copy()
# Remove target
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),
     make_column_selector(dtype_include=object)),
)

# standardizes numerical features and categorical features are one-hot encoded
X = preprocessor.fit_transform(X)
# log transform target to normalize the distribution instead of standardizing
y = np.log(y) 

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

# Create network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

# optimizer and loss function
model.compile(
    optimizer="adam",
    loss="mae",
)

# Feed the optimizer 256 rows of training data and do it 10 times
# batch_size = 256
# epoch = 10
history = model.fit(
    X, y,
    batch_size=256,
    epochs=10,
)

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df.loc[5:, ['loss']].plot()
plt.show()