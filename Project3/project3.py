from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import argparse

# parse for demo mode or not
globalDemoMode: bool = True
globalGraphMode: bool = True


parser = argparse.ArgumentParser()

parser.add_argument('--demo', action='store_true')
parser.add_argument('--graph', action='store_true')

args = parser.parse_args()

print(vars(args))

globalDemoMode = vars(args)['demo']
globalGraphMode = vars(args)['graph']

# load dataset
cancer_data = load_breast_cancer()

# and now create a dataframe from it that we control
cancer = pd.DataFrame(np.c_[cancer_data['data'], cancer_data['target']], columns=np.append(
    cancer_data['feature_names'], ['target']))

# print statistical summary of the data
print('mean: ', cancer.mean())
print('median: ', cancer.median())
print('mode: ', cancer.mode())
print('standard dev: : ', cancer.std())


# chart options - long, uncomment the ones you want to see


if globalGraphMode is True:
    # pairplot - slow, uncomment for drastic performance shift
    pair = sns.pairplot(cancer, hue='target', vars=[
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
    plt.show()

    # heatmap - not as bad
    plt.figure(figsize=(20, 20))
    sns.heatmap(cancer.corr(), annot=True, cmap='coolwarm', linewidths=2)
    plt.show()

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
    def __init__(self, weight, accuracy, solver, activation, alpha):
        self.weight = weight
        self.accuracy = accuracy
        self.solver = solver
        self.activation = activation
        self.alpha = alpha

    def __repr__(self):
        return f"{self.weight}, activation: {self.activation}, solver: {self.solver}, accuracy: {self.accuracy}, alpha: {self.alpha}"

    def __getitem__(self, i):
        return f"{i}"


alphas = [.1, .01]

activations = ['identity', 'logistic', 'tanh', 'relu']

solvers = ['lbfgs', 'sgd', 'adam']


# we want to be able to store the best weights, ordered from ascending accuracy
bestWeights: [Parameters] = []

if not globalDemoMode:
    print("Layer Tests")

    weights = []

    print("Single Hidden Layer Model Tests")

    for solver in solvers:
        for activation in activations:
            for alpha in alphas:
                for i in range(1, 20, 1):
                    x = Parameters((i,), 0, solver, activation, alpha)
                    classifier.hidden_layer_sizes = x.weight
                    classifier.activation = x.activation
                    classifier.solver = x.solver
                    classifier.fit(X_train_sc, y_train)
                    classifier.alpha = x.alpha
                    x.accuracy = classifier.score(X_test_sc, y_test)
                    print(repr(x))
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
        print(repr(item))

    print("Double Hidden Layer Tests")
    for solver in solvers:
        for activation in activations:
            for alpha in alphas:
                for i in range(1, 20, 1):
                    for j in range(1, 20, 1):
                        x = Parameters((i, j), 0, solver, activation, alpha)
                        classifier.hidden_layer_sizes = x.weight
                        classifier.activation = x.activation
                        classifier.solver = x.solver
                        classifier.alpha = x.alpha
                        classifier.fit(X_train_sc, y_train)
                        x.accuracy = classifier.score(X_test_sc, y_test)
                        print(repr(x))
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
        print(repr(item))

    print("Triple Hidden Layer Tests")
    for solver in solvers:
        for activation in activations:
            for alpha in alphas:
                for i in range(1, 20, 1):
                    for j in range(1, 20, 1):
                        for k in range(1, 20, 1):
                            x = Parameters((i, j, k), 0, solver,
                                           activation, alpha)
                            classifier.hidden_layer_sizes = x.weight
                            classifier.activation = x.activation
                            classifier.solver = x.solver
                            classifier.alpha = x.alpha
                            classifier.fit(X_train_sc, y_train)
                            x.accuracy = classifier.score(X_test_sc, y_test)
                            print(repr(x))
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
        print(repr(item))

else:
    # weights listed so I can make the confusion matricies and not run this all over again
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
    tempWeights.append(Parameters(
        (4, 12), 0.986013986013986, 'adam', 'identity', 0.0001))
    tempWeights.append(Parameters(
        (4, 12), 0.986013986013986, 'sgd', 'relu', 0.0001))
    tempWeights.append(Parameters(
        (4, 5), 0.986013986013986, 'sgd', 'relu', 0.0001))
    tempWeights.append(Parameters(
        (18, 14), 0.986013986013986, 'sgd', 'tanh', 0.0001))
    tempWeights.append(Parameters(
        (4, 18), 0.986013986013986, 'sgd', 'tanh', 0.0001))
    tempWeights.append(Parameters(
        (4, 7), 0.986013986013986, 'sgd', 'tanh', 0.0001))
    tempWeights.append(Parameters(
        (9, 4), 0.986013986013986, 'sgd', 'identity', 0.0001))
    tempWeights.append(Parameters(
        (4, 8), 0.986013986013986, 'sgd', 'identity', 0.0001))
    tempWeights.append(Parameters(
        (4, 5), 0.986013986013986, 'sgd', 'identity', 0.0001))
    tempWeights.append(Parameters(
        (16,), 0.986013986013986, 'adam', 'identity', 0.0001))

# nonsaved weights side
if globalDemoMode:
    for parameter in tempWeights:
        print("hello")
        # generate heatmaps
else:
    # saved weights Side
    for parameter in bestWeights:
        print("hello")
        # generate heatmaps


def generateConfusion(models: [Parameters]):
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for model in models:
        classifier.activation = model.activation
        classifier.hidden_layer_sizes = model.weight
        classifier.solver = model.solver
        classifier.alpha = model.alpha
        classifier.fit(X_train_sc, y_train)
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(
                classifier, X_test_sc, y_test, display_labels=['setosa', 'virginica', 'versicolor'], normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
