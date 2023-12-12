import numpy as np

#Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV


RANDOM_SED: int = 42

ml_classifiers_params = {

    'Decision Tree': {

                        'classifier': DecisionTreeClassifier(random_state=RANDOM_SED),
                        'params': {
                                    'criterion': ['gini', 'entropy'],
                                    'splitter': ['best','random'],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 5, 10]
                        }
    },

    'SVM': {

                        'classifier': SVC(random_state=RANDOM_SED),
                        'params': {
                                    'tol': [0.001, 0.0001, 0.00001],
                                    'C': [1.0, 1.5, 2.0],
                                    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
                        }
    },
    
    'Nearest Neighbors': {

                        'classifier': KNeighborsClassifier(),
                        'params': {
                                    'n_neighbors': [2, 5, 10, 20],
                                    'p': [1, 2],
                                    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                        }
    },

    'Random Forest': {

                        'classifier': RandomForestClassifier(random_state=RANDOM_SED),
                        'params': {
                                    'criterion': ['gini', 'entropy'],
                                    #'splitter': ['best','random'],
                                    'min_samples_split': [10, 40, 100, 150],
                                    'min_samples_leaf': [1, 5, 10]
                        }  
    },

    'Gaussian Naive Bayes': {

                        'classifier': GaussianNB(),
                        'params': {
                                    'var_smoothing' : np.logspace(0,-9, num=100)
                        }
    },

    'Logistic Regression': {

                        'classifier': LogisticRegression(random_state=RANDOM_SED),
                        'params': {
                                    'tol': [0.0001, 0.00001, 0.000001],
                                    'max_iter': [1000, 10000],
                                    'C': [1.0, 1.5, 2.0],
                                    'solver': ['lbfgs', 'sag', 'saga']
                        }
    },
    
    'Extra Trees': {

                        'classifier': ExtraTreesClassifier(random_state=RANDOM_SED),
                        'params': {
                                    'criterion': ['gini', 'entropy'],
                                    'n_estimators': [10, 50, 100, 200],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 5, 10]
                        }
    },


    'MLP': {

                        'classifier': MLPClassifier(random_state=RANDOM_SED),
                        'params': {
                                    'activation': ['relu', 'logistic', 'tanh'],
                                    'solver': ['adam', 'sgd'],
                                    'batch_size': [8, 16, 32, 64]
                        }
    }   

} 