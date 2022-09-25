import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Neural Networks
df = pd.read_csv("winequality-red.csv")
df.dropna()
le = LabelEncoder()
x = df.drop(['quality'] , axis = 1)
y = le.fit_transform(df.iloc[: , -1])
y = pd.DataFrame(y.reshape(len(y),1))

from imblearn.over_sampling import SMOTE
strategy = {0:1000, 1:1000, 2:1700, 3:1700, 4:1700, 5:1700}
oversample = SMOTE(sampling_strategy=strategy)
x, y = oversample.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

y_train_cat = tf.keras.utils.to_categorical(y_train, 6)
y_test_cat = tf.keras.utils.to_categorical(y_test, 6)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

model = tf.keras.models.Sequential(layers = None , name = None)
model.add(tf.keras.layers.Input(shape = 11,))
model.add(tf.keras.layers.Dense(units = 1024, activation = "relu" ))
model.add(tf.keras.layers.Dense(units = 1024, activation = "relu" ))
model.add(tf.keras.layers.Dense(units = 6, activation = "sigmoid"))

model.summary()
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics= ['accuracy'])
history = model.fit(x_train, y_train_cat,  batch_size= 32, epochs = 150 , validation_data = (x_test,y_test_cat))

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

