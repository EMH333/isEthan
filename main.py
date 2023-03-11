import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

import data
from data import get_combinations

import pickle
import gzip
from collections import defaultdict

def col_type():
    return np.int16

col_types = defaultdict(col_type)
col_types['isEthan'] = 'bool'


def get_data():
    col = ['isEthan']
    col.extend(get_combinations())
    csv = pd.read_csv('data/grams.data.gz', names=col, sep=',', compression='gzip', dtype=(col_types))
    return csv


if __name__ == '__main__':
    scaler = StandardScaler()
    df = get_data()
    X = df.drop('isEthan', axis=1)
    y = df[['isEthan']].values.flatten()
    #print(y.values)
    #actual_y = list()
    #for xy in y.values:
    #    actual_y.append(xy[0])
    
    print("Number of features before PCA: " + str(len(X.columns)))
    print("Starting to run Variance, PCA and scaler on model...")

    # remove features with low variance
    sel = VarianceThreshold(threshold=(0.01))
    X = sel.fit_transform(X.values)

    pca = PCA(n_components=0.98)
    X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # make sure we are scaled correctly
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #print the length of the test and training datasets
    print("Length of training dataset: " + str(len(X_train)))
    print("Length of test dataset: " + str(len(X_test)))

    # print number of features
    print("Number of features: " + str(len(X_train[0])))

    # rf_clf = RandomForestClassifier(criterion='entropy')
    rf_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                           hidden_layer_sizes=(70, 7), random_state=1)
    rf_clf.fit(X_train, y_train)
    y_predict = rf_clf.predict(X_test)
    print("Accuracy score of " + str(accuracy_score(y_test, y_predict)))

    # store the model
    pickle.dump(rf_clf, gzip.open('model/model.pkl.gz', 'wb'))
    pickle.dump(scaler, gzip.open('model/scaler.pkl.gz', 'wb'))
    pickle.dump(pca, gzip.open('model/pca.pkl.gz', 'wb'))
    pickle.dump(sel, gzip.open('model/sel.pkl.gz', 'wb'))
