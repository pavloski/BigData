# import load_iris function from datasets module
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# get data
iris = load_iris()

# get feature values
iris_data = iris['data']

# get target values
iris_target = iris['target']

# get feature names
iris_names = iris['feature_names']

# create a DataFrame
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

def quick_analysis (df):
    print('\033[1m' + 'Data types of the dataset: ' + '\033[0m')
    df.info()
    print('\n')
    print('\033[1m' + 'Number of Rows: ' + '\033[0m' + str(df.shape[0]) + '\n')
    print('\033[1m' + 'Number of Columns: ' + '\033[0m' + str(df.shape[1]) + '\n')
    print('\033[1m' + 'Index Range: ' + '\033[0m' + str(len(df.index)) + '\n')
    print('\033[1m' + 'Column names: ' + '\033[0m', end="")
    for name in df.columns:
        print(name +', ', end="")
    #number of null values
    nv = df.isnull().sum().sum()
    print('\n')
    print('\033[1m' + 'Number of null values: ' + '\033[0m' + str(nv) + '\n')
    print('\033[1m' + 'Mean values: ' + '\033[0m' + str(np.mean(df)) + '\n')
    print('\033[1m' + 'Standard deviations: ' + '\033[0m' + str(np.std(df)) + '\n')
    print('\033[1m' + 'Absolute Average Deviation (AAD): ' + '\033[0m' + str(df.mad()) + '\n')

quick_analysis (iris_df)

# create a box plot of attributes for all species in the dataset
iris_df.plot(kind='box')
plt.show()