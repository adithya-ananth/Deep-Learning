#5.2

'''
Batchnorm: A batch normalization layer looks at each batch as it comes in, first normalizing 
the batch with its own mean and standard deviation, and then also putting the data on a new scale 
with two trainable rescaling parameters. Batchnorm, in effect, performs a kind of coordinated 
rescaling of its inputs.

Most often, batchnorm is added as an aid to the optimization process. 
Models with batchnorm tend to need fewer epochs to complete training. 
Moreover, batchnorm can also fix various problems that can cause the training to get "stuck".
'''

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

concrete = pd.read_csv(r'CSV files\concrete.csv')
df = concrete.copy()

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

X_train = df_train.drop('CompressiveStrength', axis=1)
X_valid = df_valid.drop('CompressiveStrength', axis=1)
y_train = df_train['CompressiveStrength']
y_valid = df_valid['CompressiveStrength']

input_shape = [X_train.shape[1]]

# The dataset has not been normalized, making it messy to work with
# Training this network on a dataset without batch normalization will fail and result 
# in an empty loss vs valid_loss graph
# The processes required to minimize the loss function either does not converge or converges to
# a very large number resulting in Minimum Validation Loss = nan
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
])

model.compile(
    # SGD is more sensitive to differences of scale
    optimizer='sgd',
    loss='mae',
    # the metric monitors training and evaluates the model
    metrics=['mae'],
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=100,
)

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
plt.show()