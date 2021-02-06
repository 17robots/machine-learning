import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
import pandas as pd

flowers = pd.read_excel('irisData.xlsx', sheet_name='irisDataset')
flowers.head()
lookup_flower_name = dict(
    zip(flowers.species.unique(), flowers.name.unique())
)

print("Model: Iris Flower Predictions")
print("Number of Attributes: 4, petal length, petal width, sepal length and sepal width")
# print(lookup_flower_name)

X = flowers[['petal_length', 'petal_width', 'sepal_length', 'sepal_width']]
Y = flowers['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X_train, c=Y_train, marker='o', s=40, hist_kwds={
                         'bins': 15}, figsize=(9, 9), cmap=cmap)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['sepal_length'], X_train['sepal_width'], X_train['petal_length'],
           c=Y_train, marker='o', s=100)
ax.set_xlabel('sepal width')
ax.set_ylabel('sepal length')
ax.set_zlabel('petal length')
plt.show()

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

knn.fit(X_train, Y_train)

# calculate the validity of the model for future predictions
knn.score(X_test, Y_test)

# lets make a prediction for a flower
flower_prediction = knn.predict([[7, 2.9, 5.7, 1.8]])
print(lookup_flower_name[flower_prediction[0]])

# calculate the sensitivity of the k-NN classification accuracy to the choice of the k parameter
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    scores.append(knn.score(X_test, Y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()

# calculate the sensitivity of the k-NN classification accuracy tp the train/test split portion
t = [.8, .7, .6, .5, .4, .3, .2]
knn = KNeighborsClassifier(n_neighbors=5)
plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=1-s)
        knn.fit(X_train, Y_train)
        scores.append(knn.score(X_test, Y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training Test Proportion (%)')
plt.ylabel('accuracy')
plt.show()
