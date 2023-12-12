import pandas as pd
import random
import os
import cv2
import csv
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

from data_ml_classifiers import ml_classifiers_params


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

np.random.seed(123)
#tf.random.set_seed(123)


def training_gridsearchcv_best_estimator(model_name, num_deep_radiomics, X_train, X_test, y_train, y_test):

    kfold = 10
    scores_results = []
    runtime_train = 0.0
    start_time_train = 0.0
    best_model_voting = []

    for clf_name, clf_params in ml_classifiers_params.items():

        clf_grid_search = GridSearchCV(estimator=clf_params['classifier'], param_grid=clf_params['params'], cv = kfold)
        
        
        print(f'[INFO] {clf_name} is trainning...')
        start_time_train = time.time()
        clf_grid_search.fit(X_train, y_train)
        
        runtime_train = time.time() - start_time_train

        scores_results.append({
                                'classifier': clf_name,
                                'best_score': clf_grid_search.best_score_,
                                'best_params': clf_grid_search.best_params_,
                                'best_estimator': clf_grid_search.best_estimator_,
                                'time': runtime_train
        })
        
        best_model = clf_grid_search.best_estimator_
        best_model_voting.append((clf_name, best_model))

        final_preds = best_model.predict(X_test)

        f1 = f1_score(y_test, final_preds)
         
        precision = precision_score(y_test, final_preds)
        recall = recall_score(y_test, final_preds)

        print(f"F1-Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")


        print(f'\t[INFO] {clf_name} is trained in  %.2f seconds\n' %(runtime_train))

        runtime_train = 0.0
        start_time_train = 0.0

    #Print INFO
    df_gs_results_params = pd.DataFrame(scores_results, columns=['classifier', 'best_score', 'best_params', 'best_estimator', 'time'])
    df_gs_results_params.to_csv(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", 'df_grid_search_results_params.csv'), index=None)

    return best_model_voting



def voting_classifier(X_train, X_test, y_train, y_test, best_model_voting):
    '''
    nome
    params:
        x - >
    '''

    voting_classifier = VotingClassifier(estimators=best_model_voting, voting='hard')
    voting_classifier.fit(X_train, y_train)
    final_preds = voting_classifier.predict(X_test)

    print('[INFO] Voting Classifier using Ensemble Learning Model')
    f1 = f1_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds)
    recall = recall_score(y_test, final_preds)

    print(f"F1-Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    #Print Info
    #Salvar
    