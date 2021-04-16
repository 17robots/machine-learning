from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix

import seaborn as sb

# read in the different csv's

print('reading in csv')
airlines = pd.read_csv('airlines.csv')
airports = pd.read_csv('airports.csv')
flights = pd.read_csv('flights.csv', low_memory=False)
print('done')
print('dropping columns')

# attributeListString = ''
# for item in flights.columns:
#     attributeListString += item + ", "

# print("Data Attributes: " + attributeListString)

# df = sb.jointplot(data=flights, x="SCHEDULED_ARRIVAL", y="ARRIVAL_TIME")
# plt.show()
# print(flights.corr())
# print(flights[flights.columns[1:]].corr()[
#       'ARRIVAL_DELAY'][:].sort_values(ascending=False))

# drop cols based on correlation analysis
flights = flights.drop(['YEAR', 'FLIGHT_NUMBER', 'AIRLINE', 'DISTANCE', 'TAIL_NUMBER', 'TAXI_OUT', 'SCHEDULED_TIME', 'DEPARTURE_TIME',
                        'WHEELS_OFF', 'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'DAY_OF_WEEK', 'TAXI_IN', 'CANCELLATION_REASON'], axis=1)

print('done')
# print('filling blank values with mean')
# mean = flights.mean()
# fill na values with the mean
# flights = flights.fillna(mean)
# print('done')
print('generating heatmap with correlation')

correlation = flights.corr()
# get a heatmap of the correlation
sb.heatmap(correlation, annot=True, cmap=cm.get_cmap('gnuplot'), fmt='g')
plt.show()
