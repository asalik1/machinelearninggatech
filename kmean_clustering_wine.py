import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Get dataset
df = pd.read_csv("winequality-red.csv")
df = df.drop('quality', axis=1)

x_std = StandardScaler().fit_transform(df)

inertias = []
for i in range(1, 10):
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=i, init='k-means++')
    # Fit model to samples
    model.fit(x_std)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(range(1,10), inertias, '-o', color='black')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title("K-Means PCA Clustering")
plt.show()


from yellowbrick.cluster import SilhouetteVisualizer

model = KMeans(n_clusters=3, init='k-means++')
model.fit(x_std)
score = silhouette_score(x_std, model.labels_, metric='euclidean')
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
    visualizer.fit(x_std)

plt.show()

z = 3
model = KMeans(n_clusters=z, init='k-means++')
model.fit(x_std)

df_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(x_std)], axis=1)
# df_kmeans.columns.values[-8:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
df_kmeans.columns.values[-z:] = ['Comp 1','Comp 2', 'Comp 3']
df_kmeans['Segment K-means'] = model.labels_

# df_kmeans['Segment'] = df_kmeans['Segment K-means'].map({   0: 'first',
#                                                             1: 'second',
#                                                             2: 'third',
#                                                             3: 'fourth',
#                                                             4: 'fifth',
#                                                             5: 'sixth',
#                                                             6: 'seventh',
#                                                             7: 'eighth'  })

df_kmeans['Segment'] = df_kmeans['Segment K-means'].map({   0: 'first',
                                                            1: 'second',
                                                            2: 'third' })

x_axis = df_kmeans['Comp 2']
y_axis = df_kmeans['Comp 1']
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(z+1)
palette_list = []
for x in range(z):
    palette_list.append(cmap(x))
sns.scatterplot(x=x_axis, y=y_axis, hue = df_kmeans['Segment'], palette = palette_list)
plt.title('Clusters Visualization by K-means')
plt.show()

