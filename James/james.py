import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score

model = Sequential()
model.add(Dense(3, input_dim=6, activation='sigmoid', use_bias=True))
adam = Adam(learning_rate=0.1)
model.compile(loss='binary_crossentropy', optimizer=adam)
model.summary()

x = np.array([[0.27, 0.24, 1, 0, 1, 0],
              [0.33, 0.44, -1, 0, 0, 1],
              [0.48, 0.98, -1, 1, 0, 0],
              [0.30, 0.29, 1, 1, 0, 0],
              [0.66, 0.65, -1, 0, 1, 0]])

y = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])

model.fit(x, y, epochs=4000, batch_size=2, verbose=1)

result = model.predict([[0.38, 0.51, -1, 0, 1, 0]])
print([f"{i:.2f}" for i in result[0]])
