#4

'''
The training loss will go down either when the model learns signal or when it learns noise. 

But the validation loss will go down only when the model learns signal. (Whatever noise the model 
learned from the training set won't generalize to new data.) So, when a model learns signal both 
curves go down, but when it learns noise a gap is created in the curves. The size of the gap tells 
you how much noise the model has learned.
'''

'''
Underfitting the training set is when the loss is not as low as it could be because the model hasn't 
learned enough signal. 
Overfitting the training set is when the loss is not as low as it could be because the model learned 
too much noise. 
The trick to training deep learning models is finding the best balance between the two.

In Keras, we include early stopping in our training through a callback. 
A 'Callback' is just a function you want run every so often while the network trains. 
The early stopping callback will run after every epoch.
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

spotify = pd.read_csv(r'CSV files\spotify.csv')

X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

# A "Grouped Split" splits the artists' songs into train data and test data
# This is to help prevent signal leakage.
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

# Popularity is scaled down from 0-100 to 0-1
y_train = y_train / 100 
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

model = keras.Sequential([
    layers.Dense(128, activation ='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
)

# Implement early stopping to stop training once overfitting starts
early_stopping = callbacks.EarlyStopping(
    # minimium amount of change to count as an improvement
    min_delta=0.001,
    # how many previous epochs to check for improvement before stopping
    patience=20, 
    # enables us to keep the model where validation loss is the lowest
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
plt.show()