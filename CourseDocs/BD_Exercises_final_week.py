# Load libraries
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print('starting')

# conduct preliminary data exploration
def quick_analysis(df):
    st = '\033[1m' + 'Data types of the dataset: ' + '\033[0m' + '\n'
    st = st + str(df.dtypes)
    st = st + '\n'
    st = st + '\033[1m' + 'Number of Rows: ' + '\033[0m' + str(df.shape[0]) + '\n'
    st = st + '\033[1m' + 'Number of Columns: ' + '\033[0m' + str(df.shape[1]) + '\n'
    st = st + '\033[1m' + 'Index Range: ' + '\033[0m' + str(len(df.index)) + '\n'
    st = st + '\033[1m' + 'Column names: ' + '\033[0m' + '\n'

    for name in df.columns:
        st = st + str(name + ', ' + '\n')

    # number of null values
    nv = df.isnull().sum().sum()
    st = st + '\n'
    st = st + '\033[1m' + 'Number of null values: ' + '\033[0m' + str(nv) + '\n'
    st = st + '\033[1m' + 'Mean values: ' + '\033[0m' + str(np.mean(df)) + '\n'
    st = st + '\033[1m' + 'Standard deviations: ' + '\033[0m' + str(np.std(df)) + '\n'
    st = st + '\033[1m' + 'Absolute Average Deviation (AAD): ' + '\033[0m' + str(df.mad()) + '\n'

    return st

import pandas
import graphviz
from sklearn import tree
import numpy as np
import pandas
import graphviz
from sklearn import tree
import numpy as np
raw_data = pandas.read_csv('/rodata/exercise01/part1.csv', index_col=None)
learn_data = raw_data.copy()
classes = raw_data['class']
del learn_data['class']
classifier = tree.DecisionTreeClassifier(criterion= 'entropy')
classifier.fit(learn_data.values, classes)
classifier.predict([[500,]])
tree.plot_tree(classifier,
               feature_names=learn_data.columns)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# a different approach is to split data into trining and test

# Split data into training and test set
features_train, features_test, target_train, target_test = train_test_split(
    learn_data.values, classes, test_size=0.2, random_state=0)

# we run the model again
classifier.fit(features_train, target_train)

# and then we test it with the test set:
classifier.score(features_test, target_test)

print(classification_report(target_test, classifier.predict(features_test)))

# plotting decision boundaries

import matplotlib as plt
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(learn_data.values, classes, clf = classifier, legand = 2, zoom_factor=0.01)
plt.show()
