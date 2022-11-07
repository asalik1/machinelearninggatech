from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Get dataset
df = pd.read_csv("titanic.csv")
# Drop unneeded column(s)
df = df.drop('Name', axis=1)
df.dropna()

le = LabelEncoder()
df.Sex = le.fit_transform(df.Sex)

# y will be the survive column
y = df.iloc[:,0]
# x will be all the columns except the 'survive' column
x = df.iloc[:,1:]

df = df.drop('Survived', axis=1)
x_std = StandardScaler().fit_transform(df)

K-MEANS 
---------------------------------------------------------------------------------------------------#
from sklearn.decomposition import PCA
print("----------------------------------------------PCA K-MEANS----------------------------------")
pca = PCA(n_components=3)
pca.fit(x_std)
pca.transform(x_std)
scores_pca = pca.transform(x_std)
model = KMeans(n_clusters=2, init='k-means++')
model.fit(scores_pca)

x_train, x_test, y_train, y_test = train_test_split(x, model.labels_, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))

cont = input("Enter any key to continue")
#-------------------------------------------------------------------------------------------------#
from sklearn.decomposition import FastICA
print("----------------------------------------------ICA K-MEANS----------------------------------")

ica = FastICA(n_components=2)
ica_result = ica.fit_transform(x_std)
model = KMeans(n_clusters=2, init='k-means++')
model.fit(ica_result)

x_train, x_test, y_train, y_test = train_test_split(x, model.labels_, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))
cont = input("Enter any key to continue")

#-------------------------------------------------------------------------------------------------#
from sklearn.random_projection import GaussianRandomProjection
print("----------------------------------------------GRP K-MEANS----------------------------------")

grp_model = GaussianRandomProjection(n_components=2)
grp_result = grp_model.fit_transform(x_std)
model = KMeans(n_clusters=2, init='k-means++')
model.fit(grp_result)

x_train, x_test, y_train, y_test = train_test_split(x, model.labels_, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))
cont = input("Enter any key to continue")

#------------------------------------------------------------------------------------------------#
from sklearn.cluster import FeatureAgglomeration
print("----------------------------------------------FEATURE AGGLOMORATION K-MEANS---------------")

lda = FeatureAgglomeration(n_clusters = 2)
lda_result = lda.fit_transform(x_std, y)
model = KMeans(n_clusters=2, init='k-means++')
model.fit(lda_result)

x_train, x_test, y_train, y_test = train_test_split(x, model.labels_, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))

#------------------------------------------------------------------------------------------------#

# Expectation Maximization
#---------------------------------------------------------------------------------------------------#
from sklearn.decomposition import PCA
print("----------------------------------------------PCA EM----------------------------------")
pca = PCA(n_components=3)
pca.fit(x_std)
pca.transform(x_std)
scores_pca = pca.transform(x_std)
model = GaussianMixture(n_components=2)
model.fit(scores_pca)
labels = model.predict(scores_pca)

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))

cont = input("Enter any key to continue")
#-------------------------------------------------------------------------------------------------#
from sklearn.decomposition import FastICA
print("----------------------------------------------ICA EM----------------------------------")
ica = FastICA(n_components=2)
ica_result = ica.fit_transform(x_std)
model = GaussianMixture(n_components=2)
model.fit(ica_result)
labels = model.predict(ica_result)

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))
cont = input("Enter any key to continue")

#-------------------------------------------------------------------------------------------------#
from sklearn.random_projection import GaussianRandomProjection
print("----------------------------------------------GRP EM----------------------------------")

grp_model = GaussianRandomProjection(n_components=2)
grp_result = grp_model.fit_transform(x_std)
model = GaussianMixture(n_components=2)
model.fit(grp_result)
labels = model.predict(grp_result)

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))
cont = input("Enter any key to continue")

#------------------------------------------------------------------------------------------------#
from sklearn.cluster import FeatureAgglomeration
print("----------------------------------------------FEATURE AGGLOMORATION EM---------------")

lda = FeatureAgglomeration(n_clusters = 2)
lda_result = lda.fit_transform(x_std, y)
model = GaussianMixture(n_components=2)
model.fit(lda_result)
labels = model.predict(lda_result)

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape = 6,))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,  batch_size= 16, epochs = 150 , validation_data = (x_test,y_test))

#------------------------------------------------------------------------------------------------#


