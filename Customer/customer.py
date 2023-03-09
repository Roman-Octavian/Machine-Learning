import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
from pickle import dump, load

# Import data locally
# Can't find file without absolute path
dataFrame = pd.read_csv('C:/Users/Octavian/PycharmProjects/machine-learning/csv/customer_staying_or_not.csv')
dataFrame.head()

# Remove missing data if any
dataFrame.isnull().sum()
dataFrame.dropna(inplace=True)

# Assign input (X) and output (y) from data frame
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 1000)
X = dataFrame.iloc[1:, 3:13]
y = dataFrame.iloc[1:, -1]

# Convert text to categorical data and numpy array
X = pd.get_dummies(X)
columnNames = list(X.columns)
X = X.values
y = y.values

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = Sequential()
model.add(Dense(26, input_dim=13, activation='relu', use_bias=True, name="Input"))
model.add(Dense(26, activation='relu', use_bias=True, name="Hidden_Layer_1"))
model.add(Dense(26, activation='relu', use_bias=True, name="Hidden_Layer_2"))
model.add(Dense(26, activation='relu', use_bias=True, name="Hidden_Layer_3"))
model.add(Dense(1, activation='sigmoid', name="Result"))
adam = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=1000, verbose=1)

# Show loss vs. epochs
loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)), y=loss)
model.evaluate(X_test, y_test, verbose=1)

# Get details about each type of prediction
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Predict new value
print(columnNames)
new_value = [[600, 40, 3, 60000, 2, 1, 1, 50000, 1, 0, 0, 0, 1]]
new_value = scaler.transform(new_value)
print("\nResult to b) part of exercise:")
print(int(model.predict(new_value)))
