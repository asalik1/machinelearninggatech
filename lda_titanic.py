import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import FeatureAgglomeration

# Get dataset
df = pd.read_csv("titanic.csv")
# Use label encoder to reclassify 'sex' category
le = LabelEncoder()
df.Sex = le.fit_transform(df.Sex)
y = df.iloc[:,0]
print(y)
# Drop unneeded column(s)
df = df.drop('Name', axis=1)
df = df.drop('Survived', axis=1)

x_std = StandardScaler().fit_transform(df)
lda = FeatureAgglomeration(n_clusters = 2)
lda_result = lda.fit_transform(x_std, y)
print(lda_result.shape)
comp_one = lda_result[:,0]
comp_two = lda_result[:,1]
# result_signal_3 = ica_result[:,2]

plt.title('Random  Projection Components')
plt.plot(comp_one, c="#df8efd")
plt.plot(comp_two, c="#87de72")
# plt.plot(result_signal_3, c="#f65e97")
plt.show()

#LDA K-Means
inertias = []
for i in range(1, 10):
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=i, init='k-means++')
    # Fit model to samples
    model.fit(lda_result)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(range(1,10), inertias, '-o', color='black')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title("K-Means FA Clustering")
plt.show()

from yellowbrick.cluster import SilhouetteVisualizer

model = KMeans(n_clusters=3, init='k-means++')
model.fit(lda_result)
score = silhouette_score(lda_result, model.labels_, metric='euclidean')
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
    visualizer.fit(lda_result)

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
model.fit(lda_result)
df_lda_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(lda_result)], axis=1)
# df_lda_kmeans.columns.values[-8:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
df_lda_kmeans.columns.values[-z:] = ['Comp 1','Comp 2']
df_lda = df_lda_kmeans['Segment K-means FA'] = model.labels_

# df_lda_kmeans['Segment'] = df_lda_kmeans['Segment K-means LDA'].map({   0: 'first',
#                                                                         1: 'second',
#                                                                         2: 'third',
#                                                                         3: 'fourth',
#                                                                         4: 'fifth',
#                                                                         5: 'sixth',
#                                                                         6: 'seventh',
#                                                                         7: 'eighth'  
#})

df_lda_kmeans['Segment'] = df_lda_kmeans['Segment K-means FA'].map({   0: 'first',
                                                                        1: 'second'
})

x_axis = df_lda_kmeans['Comp 2']
y_axis = df_lda_kmeans['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_lda_kmeans['Segment'], palette = palette_list)
plt.title('Clusters Visualization by FA K-means')
plt.show()

# FA EM
from sklearn.mixture import GaussianMixture
z = 7
k_arr = np.arange(9) + 1
models = [
 GaussianMixture(n_components=k).fit(lda_result)
 for k in k_arr
]

# Compute metrics to determine best hyperparameter
AIC = [m.aic(lda_result) for m in models]
BIC = [m.bic(lda_result) for m in models]
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
model.fit(lda_result)
	
labels = model.predict(lda_result)
plt.figure(figsize=(9,7))

df_em = pd.concat([df.reset_index(drop = True), pd.DataFrame(lda_result)], axis=1)
# df_em.columns.values[-z:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
# Can only go up to 9
df_em.columns.values[-z:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4', 'Comp 5', 'Comp 6', 'Comp 7']
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
                                                    5: 'sixth',
                                                    6: 'seventh'
})

x_axis = df_em['Comp 2']
y_axis = df_em['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_em['Segment'], palette = palette_list)
plt.title('Clusters Visualization by FA EM')
plt.show()