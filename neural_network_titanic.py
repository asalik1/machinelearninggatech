import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import tensorflow as tf

# Neural Networks
# Get dataset
df = pd.read_csv("titanic.csv")
# Drop unneeded column(s)
df = df.drop('Name', axis=1)
df.dropna()

# target = df.pop('Survived')
le = LabelEncoder()
df.Sex = le.fit_transform(df.Sex)
# x will be all the columns except the 'survive' column
x = df.iloc[:,1:]
# y will be the survive column
y = df.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))

plt.plot(history.history['accuracy'], label='Accuracy training data')
plt.plot(history.history['val_accuracy'], label='Accuracy validation data')
plt.legend()
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'], label='Loss training data')
plt.plot(history.history['val_loss'], label='Loss validation data')
plt.legend()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
