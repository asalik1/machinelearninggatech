import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
from matplotlib import pyplot as plt

#Get dataset
df = pd.read_csv("titanic.csv")
# Use label encoder to reclassify 'sex' category
le = LabelEncoder()
df.Sex = le.fit_transform(df.Sex)
# Drop unneeded column(s)
df = df.drop('Name', axis=1)
df = df.drop('Survived', axis=1)
x_std = StandardScaler().fit_transform(df)

grp_model = GaussianRandomProjection(n_components=2)
grp_result = grp_model.fit_transform(x_std)
print(grp_result.shape)

# comp_one = grp_result[:,0]
# comp_two = grp_result[:,1]

# plt.title('Random  Projection Components')
# plt.plot(comp_one, c="#df8efd")
# plt.plot(comp_two, c="#87de72")
# # plt.plot(result_signal_3, c="#f65e97")
# plt.show()

#GRP K-Means
inertias = []
for i in range(1, 10):
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=i, init='k-means++')
    # Fit model to samples
    model.fit(grp_result)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(range(1,10), inertias, '-o', color='black')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title("K-Means GRP Clustering")
plt.show()

from yellowbrick.cluster import SilhouetteVisualizer

model = KMeans(n_clusters=3, init='k-means++')
model.fit(grp_result)
score = silhouette_score(grp_result, model.labels_, metric='euclidean')
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
    visualizer.fit(grp_result)

plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

z = 2
cmap = get_cmap(z+1)
palette_list = []
for x in range(z):
    palette_list.append(cmap(x))

model = KMeans(n_clusters=z, init='k-means++')
model.fit(grp_result)
df_grp_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(grp_result)], axis=1)
# df_grp_kmeans.columns.values[-8:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
df_grp_kmeans.columns.values[-z:] = ['Comp 1','Comp 2']
df_grp = df_grp_kmeans['Segment K-means GRP'] = model.labels_

# df_grp_kmeans['Segment'] = df_grp_kmeans['Segment K-means GRP'].map({   0: 'first',
#                                                                         1: 'second',
#                                                                         2: 'third',
#                                                                         3: 'fourth',
#                                                                         4: 'fifth',
#                                                                         5: 'sixth',
#                                                                         6: 'seventh',
#                                                                         7: 'eighth'  
#})

df_grp_kmeans['Segment'] = df_grp_kmeans['Segment K-means GRP'].map({   0: 'first',
                                                                        1: 'second'
})

x_axis = df_grp_kmeans['Comp 2']
y_axis = df_grp_kmeans['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_grp_kmeans['Segment'], palette = palette_list)
plt.title('Clusters Visualization by GRP K-means')
plt.show()

# GRP EM
from sklearn.mixture import GaussianMixture
z = 6
k_arr = np.arange(9) + 1
models = [
 GaussianMixture(n_components=k).fit(grp_result)
 for k in k_arr
]

# Compute metrics to determine best hyperparameter
AIC = [m.aic(grp_result) for m in models]
BIC = [m.bic(grp_result) for m in models]
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
model.fit(grp_result)
	
labels = model.predict(grp_result)
plt.figure(figsize=(9,7))

df_em = pd.concat([df.reset_index(drop = True), pd.DataFrame(grp_result)], axis=1)
# df_em.columns.values[-z:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
# Can only go up to 9
df_em.columns.values[-z:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4', 'Comp 5', 'Comp 6']
df_em['Segment K-means'] = labels

# df_em['Segment'] = df_em['Segment K-means'].map({   0: 'first',
#                                                     1: 'second',
#                                                     2: 'third',
#                                                     3: 'fourth',
#                                                     4: 'fifth',
#                                                     5: 'sixth',
#                                                     6: 'seventh',
#                                                     7: 'eighth'  })

df_em['Segment'] = df_em['Segment K-means'].map({   0: 'first',
                                                    1: 'second',
                                                    2: 'third',
                                                    3: 'fourth',
                                                    4: 'fifth',
                                                    5: 'sixth'
})

x_axis = df_em['Comp 2']
y_axis = df_em['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_em['Segment'], palette = palette_list)
plt.title('Clusters Visualization by GRP EM')
plt.show()