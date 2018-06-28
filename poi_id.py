#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import numpy as np
import pandas as pd

#feature selection
features_list = ['poi',
                 'salary',
                 'bonus',
                 'long_term_incentive',
                 'deferred_income',
                 'deferral_payments',
                 'loan_advances',
                 'other',
                 'expenses',
                 'director_fees',
                 'total_payments',
                 'exercised_stock_options',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'from_messages',
                 'from_poi_to_this_person',
                 'to_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

#Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame(data = data_dict).T[features_list]

#Changing null-values to zero
df.iloc[:,:15] = df.iloc[:,:15].replace('NaN', 0)
#Changing null-values to median
df.iloc[:,15:] = df.iloc[:,15:].replace('NaN', (df.iloc[:,15:].median()).astype(int))

#Create new features
df['poi_email_to'] = df['from_this_person_to_poi'] / (df['to_messages']+1)
df['poi_email_from'] = df['from_poi_to_this_person'] / (df['from_messages']+1)
df['liquidity'] = (df.iloc[:,[0,3,5,6,10,16]].sum(axis = 1)) / (df['total_payments'] + df['total_stock_value']+1)

#Remove outliers
df = df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'])
df = df.T.replace(np.nan, 0)

### Extract features and labels from dataset for local testing
my_dataset = df
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Create classifier using pipeline & optimized features
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA

combined_features = FeatureUnion([("pca", PCA(n_components = 2)), ("less_dim", SelectKBest(k = 'all'))])

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_select', combined_features),
    ('SVC', SVC(gamma = 0.1, C = 100, degree = 2, kernel = 'poly'))
])

### Record testable details
dump_classifier_and_data(clf, my_dataset, features_list)