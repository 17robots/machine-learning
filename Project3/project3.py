from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

# load dataset
cancer_data = load_breast_cancer()

# and now create a dataframe from it that we control
cancer = pd.DataFrame(np.c_[cancer_data['data'], cancer_data['target']], columns=np.append(
    cancer_data['feature_names'], ['target']))

# print statistical summary of the data
print(cancer.describe())


# chart options - long, uncomment the ones you want to see

# pairplot - slow, uncomment for drastic performance shift
# pair = sns.pairplot(cancer, hue='target', vars=[
# 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
# plt.show()

# heatmap - not as bad
# plt.figure(figsize=(20, 20))
# sns.heatmap(cancer.corr(), annot=True, cmap='coolwarm', linewidths=2)
# plt.show()

# split into train test sets
X = cancer.drop(['target'], axis=1)
print(X.head())

y = cancer['target']
print(y.head())

X_train, X_ttest, y_train, y_test = train_test_split(
    X, y, test_size=.25, random_state=1)

# scale the data for our model
sc = StandardScaler().fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_ttest)


classifier = MLPClassifier(random_state=1)  # keeping random state 1
classifier.max_iter = sys.maxsize * sys.maxsize

# architecture tests


class Parameters:
    def __init__(self, weight, accuracy, solver, activation):
        self.weight = weight
        self.accuracy = accuracy
        self.solver = solver
        self.activation = activation

    def __getitem__(self, i):
        return f"{i}"


activations = ['identity', 'logistic', 'tanh', 'relu']

solvers = ['sgd', 'adam']


# we want to be able to store the best weights, ordered from ascending accuracy
bestWeights: [Parameters] = []

print("Layer Tests")

weights = []

print("Single Hidden Layer Model Tests")

for solver in solvers:
    for activation in activations:
        for i in range(1, 20, 1):
            x = Parameters((i,), 0, solver, activation)
            classifier.hidden_layer_sizes = x.weight
            classifier.activation = x.activation
            classifier.solver = x.solver
            classifier.fit(X_train_sc, y_train)
            x.accuracy = classifier.score(X_test_sc, y_test)
            print(
                f"{x.weight}, activation: {x.activation}, solver: {x.solver}, accuracy: {x.accuracy}")
            if len(bestWeights) < 10:
                bestWeights.append(x)
                bestWeights.sort(key=lambda x: x[3])
            else:
                for j in range(9, -1, -1):
                    if bestWeights[j].accuracy < x.accuracy:
                        bestWeights[j] = x
                        break

print("Best Parameters after 1 layer: ")
for item in bestWeights:
    print(f"{item.weight}, activation: {item.activation}, solver: {item.solver}, accuracy: {item.accuracy}")

print("Double Hidden Layer Tests")
for solver in solvers:
    for activation in activations:
        for i in range(1, 20, 1):
            for j in range(1, 20, 1):
                x = Parameters((i, j), 0, solver, activation)
                classifier.hidden_layer_sizes = x.weight
                classifier.activation = x.activation
                classifier.solver = x.solver
                classifier.fit(X_train_sc, y_train)
                x.accuracy = classifier.score(X_test_sc, y_test)
                print(
                    f"{x.weight}, activation: {x.activation}, solver: {x.solver}, accuracy: {x.accuracy}")
                if len(bestWeights) < 10:
                    bestWeights.append(x)
                    bestWeights.sort(key=lambda x: x[3])
                else:
                    for k in range(9, -1, -1):
                        if bestWeights[k].accuracy < x.accuracy:
                            bestWeights[k] = x
                            break

print("Best Parameters after 2 layers: ")
for item in bestWeights:
    print(f"{item.weight}, activation: {item.activation}, solver: {item.solver}, accuracy: {item.accuracy}")

print("Triple Hidden Layer Tests")
for solver in solvers:
    for activation in activations:
        for i in range(1, 20, 1):
            for j in range(1, 20, 1):
                for k in range(1, 20, 1):
                    x = Parameters((i, j, k), 0, solver, activation)
                    classifier.hidden_layer_sizes = x.weight
                    classifier.activation = x.activation
                    classifier.solver = x.solver
                    classifier.fit(X_train_sc, y_train)
                    x.accuracy = classifier.score(X_test_sc, y_test)
                    print(
                        f"{x.weight}, activation: {x.activation}, solver: {x.solver}, accuracy: {x.accuracy}")
                    if len(bestWeights) < 10:
                        bestWeights.append(x)
                        bestWeights.sort(key=lambda x: x[3])
                    else:
                        for l in range(9, -1, -1):
                            if bestWeights[l].accuracy < x.accuracy:
                                bestWeights[l] = x
                                break


print("Best Parameters after 3 layers: ")
for item in bestWeights:
    print(f"{item.weight}, activation: {item.activation}, solver: {item.solver}, accuracy: {item.accuracy}")

# weights listed so I can make the heatmaps and not run this all over again
tempWeights = []


'''
(4, 12), activation: identity, solver: adam, accuracy: 0.986013986013986
(4, 12), activation: relu, solver: sgd, accuracy: 0.986013986013986
(4, 5), activation: relu, solver: sgd, accuracy: 0.986013986013986
(18, 14), activation: tanh, solver: sgd, accuracy: 0.986013986013986
(4, 18), activation: tanh, solver: sgd, accuracy: 0.986013986013986
(4, 7), activation: tanh, solver: sgd, accuracy: 0.986013986013986
(9, 4), activation: identity, solver: sgd, accuracy: 0.986013986013986
(4, 8), activation: identity, solver: sgd, accuracy: 0.986013986013986
(4, 5), activation: identity, solver: sgd, accuracy: 0.986013986013986
(16,), activation: identity, solver: adam, accuracy: 0.986013986013986
'''
tempWeights.append(Parameters((4, 12), 0.986013986013986, 'adam', 'identity'))
tempWeights.append(Parameters((4, 12), 0.986013986013986, 'sgd', 'relu'))
tempWeights.append(Parameters((4, 5), 0.986013986013986, 'sgd', 'relu'))
tempWeights.append(Parameters((18, 14), 0.986013986013986, 'sgd', 'tanh'))
tempWeights.append(Parameters((4, 18), 0.986013986013986, 'sgd', 'tanh'))
tempWeights.append(Parameters((4, 7), 0.986013986013986, 'sgd', 'tanh'))
tempWeights.append(Parameters((9, 4), 0.986013986013986, 'sgd', 'identity'))
tempWeights.append(Parameters((4, 8), 0.986013986013986, 'sgd', 'identity'))
tempWeights.append(Parameters((4, 5), 0.986013986013986, 'sgd', 'identity'))
tempWeights.append(Parameters((16,), 0.986013986013986, 'adam', 'identity'))

# nonsaved weights side
for parameter in tempWeights:
    print("hello")


# saved weights Side
for parameter in bestWeights:
    print("hello")
