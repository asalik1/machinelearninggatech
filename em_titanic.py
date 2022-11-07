import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Get dataset
df = pd.read_csv("titanic.csv")
# Use label encoder to reclassify 'sex' category
le = LabelEncoder()
df.Sex = le.fit_transform(df.Sex)
# Drop unneeded column(s)
df = df.drop('Name', axis=1)
df = df.drop('Survived', axis=1)
print(df)
x_std = StandardScaler().fit_transform(df)

k_arr = np.arange(9) + 1
models = [
 GaussianMixture(n_components=k).fit(x_std)
 for k in k_arr
]

# Compute metrics to determine best hyperparameter
AIC = [m.aic(x_std) for m in models]
BIC = [m.bic(x_std) for m in models]
# Plot these metrics
plt.plot(k_arr, AIC, label='AIC')
plt.plot(k_arr, BIC, label='BIC')
plt.xlabel('Number of Components ($k$)')
plt.legend()
plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

z = 8
cmap = get_cmap(z)
palette_list = []
for x in range(z):
    palette_list.append(cmap(x))


model = GaussianMixture(n_components=z)
model.fit(x_std)
labels = model.predict(x_std)
plt.figure(figsize=(9,7))

df_em = pd.concat([df.reset_index(drop = True), pd.DataFrame(x_std)], axis=1)
df_em.columns.values[-z:] = ['Comp 1','Comp 2', 'Comp 3','Comp 4','Comp 5','Comp 6', 'Comp 7', 'Comp 8']
df_em['Segment K-means'] = labels

df_em['Segment'] = df_em['Segment K-means'].map({   0: 'first',
                                                    1: 'second',
                                                    2: 'third',
                                                    3: 'fourth',
                                                    4: 'fifth',
                                                    5: 'sixth',
                                                    6: 'seventh',
                                                    7: 'eighth'   })

x_axis = df_em['Comp 2']
y_axis = df_em['Comp 1']
sns.scatterplot(x=x_axis, y=y_axis, hue = df_em['Segment'], palette = palette_list)
plt.title('Clusters Visualization by EM')
plt.show()

