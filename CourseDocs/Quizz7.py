import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read in the data and assign the columns names in accordance with the 'spambase.names' file
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
spam_df = pd.read_csv(url, sep = ',',header= None)

#read names using pd.read that yields a dataframe
url= 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names'
file_names = pd.read_table(url)
file_names.columns = ['c1']
names =list()
count = 0

# iterate across all rows of the dataframe
for line in file_names.c1:
    line.strip()
    count = count + 1
    if line.startswith('word') or line.startswith('char_') or line.startswith('capital_') :
        words =line.split()
        names.append(words[0].rstrip(':'))
        count = count + 1

# finally we add the target column 'spam' and pass it to the dataframe to make the column headers
names.append('spam')
spam_df.columns = names

# looking at data sample
spam_df.sample(5)


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


print(quick_analysis(spam_df))

# Answer here

# Plot histogram
sns.set_theme(color_codes=True)
spam_df.spam = spam_df.spam.astype('category')
ax = sns.histplot(data=spam_df.spam, palette="Set2")
ax.set(title='Histogram showing class distribution in the spambase dataset')
ax.set(xlabel='Spam class')
ax.set(ylabel='Number of occurrences of a class')
ax.set_xticks(range(0,2))

# plt.xlabel('Spam class')
# plt.ylabel('Number of occurrences of a class')
# plt.title('Histogram showing class distribution in the spambase dataset')

spam_df.spam = spam_df.spam.astype('int64')
sp = spam_df.spam.sum()/spam_df.shape[0]
nsp = 1-sp
print('The class distribution of the spambase dataset (%spam, %non-spam) is ({0:.2f}%, {1:.2f}%).'.format(sp*100, nsp*100))

# Answer here
corrMatrix = spam_df.corr()

# if we want to see the plot...
#sns.set(rc={"figure.figsize":(15, 15)})
#sns.heatmap(corrMatrix, annot=True)
#plt.show()

# first compute the absolute value of correlation coefficients in'corrMatrix.spam'
#then sort the panda series on descending order
max_corr_feature = np.abs(corrMatrix.spam).sort_values(ascending=False).index[1]

print('The feature with the highest correlation with the spam_class is "{}"'.format(max_corr_feature))

# Answer here

# import seaborn as sns

# create a subset of the datframe
cols = [51,52, 53,57]
spam_df2 = spam_df[spam_df.columns[cols]]


sns.pairplot(spam_df2, hue='spam', height=4)
# Plot pairplot

# Answer here

# Load libraries
from sklearn.dummy import DummyClassifier
# see the intro to sklearn in the resources folder
# the data structures are important here

dummy_clf = DummyClassifier(strategy="most_frequent")
features = spam_df.drop('spam' , axis =1).to_numpy()
target = spam_df.spam.to_numpy()

dummy_clf.fit(features, target)

num_spam_predictions=0
count = 0
for i in range(1,1001):

    #generate a random email
    test = np.random.rand(1,57)

    # make the dummy prediction
    prediction = dummy_clf.predict(test)
    num_spam_predictions = num_spam_predictions + prediction
    count = i

print ('After "{}" simulations, the numer of emails classified as spam are "{}"'.format(count, int(num_spam_predictions)))

# Answer here

# Load libraries
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from IPython.display import Image
from sklearn import tree

# Create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0)

# Train model
model = decisiontree.fit(features, target)

# Create DOT data
dot_data = tree.export_graphviz(decisiontree,
                                out_file=None,
                                feature_names=spam_df.columns[0:57],
                                class_names=spam_df.columns[57])

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
#Image(graph.create_png())
Image(graph.create_jpeg())

