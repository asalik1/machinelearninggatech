import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Get dataset
df = pd.read_csv("winequality-red.csv")
df = df.drop('quality', axis=1)

x_std = StandardScaler().fit_transform(df)
# Create a PCA instance
pca = PCA(n_components=11)
principalComponents = pca.fit_transform(x_std)
# Plot the explained variances
features = range(pca.n_components_)
plt.plot(features, pca.explained_variance_ratio_.cumsum(), color='black', marker='o')
plt.xlabel('PCA features')
plt.ylabel('Cumulative variance %')
plt.xticks(features)
plt.show()
plt.clf()

pca = PCA(n_components=4)
pca.fit(x_std)
pca.transform(x_std)
scores_pca = pca.transform(x_std)
print(pca.components_)

#PCA K-Means
inertias = []
for i in range(1, 10):
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=i, init='k-means++')
    # Fit model to samples
    model.fit(scores_pca)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(range(1,10), inertias, '-o', color='black')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title("K-Means PCA Clustering")
plt.show()

from yellowbrick.cluster import SilhouetteVisualizer

model = KMeans(n_clusters=4, init='k-means++')
model.fit(scores_pca)
score = silhouette_score(scores_pca, model.labels_, metric='euclidean')
print('Silhouetter Score: %.3f' % score)

fig, ax = plt.subplots(4, 2, figsize=(15,8))
for i in [2, 3, 4, 5, 6, 7, 8, 9]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(scores_pca)

plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

z = 4
cmap = get_cmap(z+1)
palette_list = []
for x in range(z):
    palette_list.append(cmap(x))

model = KMeans(n_clusters=z, init='k-means++')
model.fit(scores_pca)
df_pca_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(scores_pca)], axis=1)
# df_pca_kmeans.columns.values[-8:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
df_pca_kmeans.columns.values[-z:] = ['Comp 1','Comp 2','Comp 3', 'Comp 4']
df_pca_kmeans['Segment K-means PCA'] = model.labels_

# df_pca_kmeans['Segment'] = df_pca_kmeans['Segment K-means PCA'].map({   0: 'first',
#                                                                         1: 'second',
#                                                                         2: 'third',
#                                                                         3: 'fourth',
#                                                                         4: 'fifth',
#                                                                         5: 'sixth',
#                                                                         6: 'seventh',
#                                                                         7: 'eighth'  })

df_pca_kmeans['Segment'] = df_pca_kmeans['Segment K-means PCA'].map({   0: 'first',
                                                                        1: 'second',
                                                                        2: 'third',
                                                                        3: 'fourth'  })


x_axis = df_pca_kmeans['Comp 2']
y_axis = df_pca_kmeans['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_pca_kmeans['Segment'], palette = palette_list)
plt.title('Clusters Visualization by PCA K-means')
plt.show()

z = 3
cmap = get_cmap(z+1)
palette_list = []
for x in range(z):
    palette_list.append(cmap(x))

model = KMeans(n_clusters=z, init='k-means++')
model.fit(scores_pca)
df_pca_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(scores_pca)], axis=1)
# df_pca_kmeans.columns.values[-8:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
df_pca_kmeans.columns.values[-z:] = ['Comp 1','Comp 2','Comp 3']
df_pca_kmeans['Segment K-means PCA'] = model.labels_

# df_pca_kmeans['Segment'] = df_pca_kmeans['Segment K-means PCA'].map({   0: 'first',
#                                                                         1: 'second',
#                                                                         2: 'third',
#                                                                         3: 'fourth',
#                                                                         4: 'fifth',
#                                                                         5: 'sixth',
#                                                                         6: 'seventh',
#                                                                         7: 'eighth'  })

df_pca_kmeans['Segment'] = df_pca_kmeans['Segment K-means PCA'].map({   0: 'first',
                                                                        1: 'second',
                                                                        2: 'third' })


x_axis = df_pca_kmeans['Comp 2']
y_axis = df_pca_kmeans['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_pca_kmeans['Segment'], palette = palette_list)
plt.title('Clusters Visualization by PCA K-means')
plt.show()

z = 2
cmap = get_cmap(z+1)
palette_list = []
for x in range(z):
    palette_list.append(cmap(x))

model = KMeans(n_clusters=z, init='k-means++')
model.fit(scores_pca)
df_pca_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(scores_pca)], axis=1)
# df_pca_kmeans.columns.values[-8:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
df_pca_kmeans.columns.values[-z:] = ['Comp 1','Comp 2']
df_pca_kmeans['Segment K-means PCA'] = model.labels_

# df_pca_kmeans['Segment'] = df_pca_kmeans['Segment K-means PCA'].map({   0: 'first',
#                                                                         1: 'second',
#                                                                         2: 'third',
#                                                                         3: 'fourth',
#                                                                         4: 'fifth',
#                                                                         5: 'sixth',
#                                                                         6: 'seventh',
#                                                                         7: 'eighth'  })

df_pca_kmeans['Segment'] = df_pca_kmeans['Segment K-means PCA'].map({   0: 'first',
                                                                        1: 'second'})


x_axis = df_pca_kmeans['Comp 2']
y_axis = df_pca_kmeans['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_pca_kmeans['Segment'], palette = palette_list)
plt.title('Clusters Visualization by PCA K-means')
plt.show()

# PCA EM
from sklearn.mixture import GaussianMixture
z = 6
k_arr = np.arange(11) + 1
models = [
 GaussianMixture(n_components=k).fit(scores_pca)
 for k in k_arr
]

# Compute metrics to determine best hyperparameter
AIC = [m.aic(scores_pca) for m in models]
BIC = [m.bic(scores_pca) for m in models]
# Plot these metrics
plt.plot(k_arr, AIC, label='AIC')
plt.plot(k_arr, BIC, label='BIC')
plt.xlabel('Number of Components ($k$)')
plt.legend()
plt.show()

cmap = get_cmap(z+1)
palette_list = []
for x in range(z):
    palette_list.append(cmap(x))

model = GaussianMixture(n_components=z)
model.fit(scores_pca)
# min 10
labels = model.predict(scores_pca)
plt.figure(figsize=(9,7))

df_em = pd.concat([df.reset_index(drop = True), pd.DataFrame(scores_pca)], axis=1)
df_em.columns.values[-z:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6']
# df_em.columns.values[-z:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
df_em['Segment K-means'] = labels

# df_em['Segment'] = df_em['Segment K-means'].map({   0: 'first',
#                                                     1: 'second',
#                                                     2: 'third',
#                                                     3: 'fourth',
#                                                     4: 'fifth',
#                                                     5: 'sixth',
#                                                     6: 'seventh',
#                                                     7: 'eighth' })

df_em['Segment'] = df_em['Segment K-means'].map({   0: 'first',
                                                    1: 'second',
                                                    2: 'third',
                                                    3: 'fourth',
                                                    4: 'fifth',
                                                    5: 'sixth'})

x_axis = df_em['Comp 2']
y_axis = df_em['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_em['Segment'], palette = palette_list)
plt.title('Clusters Visualization by PCA EM')
plt.show()
