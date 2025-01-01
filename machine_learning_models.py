import os
import time
import numpy as np
import pandas as pd

#Classifiers and GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Classifiers with parameters used in GridSearchCV
from data_ml_classifiers import ml_classifiers_params


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


RANDOM_SED: int = 42

np.random.seed(RANDOM_SED)


def training_gridsearchcv_best_estimator(model_name, num_deep_radiomics, X_train, X_test, y_train, y_test) -> list:
    """ Training GridSearchCV with best estimator for each classifier

    Args:
        model_name (_type_): _description_
        num_deep_radiomics (_type_): _description_
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_

    Returns:
        _type_: _description_
    """  

    kfold = 10
    runtime_train = 0.0
    start_time_train = 0.0
    scores_results = []
    gridsearch_results = []
    best_model_voting = []

    for clf_name, clf_params in ml_classifiers_params.items():

        clf_grid_search = GridSearchCV(estimator=clf_params['classifier'], param_grid=clf_params['params'], cv=kfold, n_jobs=-1)
        
        print(f'[INFO] {clf_name} is trainning...')
        start_time_train = time.time()
        clf_grid_search.fit(X_train, y_train)
        
        best_model = clf_grid_search.best_estimator_ 
        best_model_voting.append((clf_name, best_model)) #Save best model for Voting Classifier

        final_preds = best_model.predict(X_test)

        #Metrics
        accuracy_score_ = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='accuracy')
        roc_auc_score_ = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='roc_auc')
        f1_score_ = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='f1_macro')
        precision_score_ = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='precision_macro')
        recall_score_ = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='recall_macro')
        classification_report_ = classification_report(y_test, final_preds)
        confusion_matrix_ = confusion_matrix(y_test, final_preds)

        runtime_train = time.time() - start_time_train

        #Append results for each classifier and parameters used in GridSearchCV    
        gridsearch_results.append({
            'classifier': clf_name,
            'best_score': clf_grid_search.best_score_,
            'best_params': clf_grid_search.best_params_,
            'best_estimator': clf_grid_search.best_estimator_,
            'time': runtime_train
        })

        #Append metrics results for each classifier with best parameters
        scores_results.append({
            'classifier': clf_name,
            'accuracy_score': accuracy_score_.mean(),
            'accuracy_score_std': accuracy_score_.std(),
            'roc_auc_score': roc_auc_score_.mean(),
            'roc_auc_score_std': roc_auc_score_.std(),
            'f1_score': f1_score_.mean(),
            'f1_score_std': f1_score_.std(),
            'precision_score': precision_score_.mean(),
            'precision_score_std': precision_score_.std(),
            'recall_score': recall_score_.mean(),
            'recall_score_std': recall_score_.std(),
            'classification_report': classification_report_,
            'confusion_matrix': confusion_matrix_,
            'time': runtime_train
        })
        
        print(f"Acurracy: {accuracy_score_.mean()} +/- {accuracy_score_.std()}")
        print(f"ROC AUC: {roc_auc_score_.mean()} +/- {roc_auc_score_.std()}")
        print(f"F1-Score: {f1_score_.mean()} +/- {f1_score_.std()}")
        print(f"Precision: {precision_score_.mean()} +/- {precision_score_.std()}")
        print(f"Recall: {recall_score_.mean()} +/- {recall_score_.std()}")
        print(f"Classification Report: \n {classification_report_}")
        print(f"Confusion Matrix: \n {confusion_matrix_}")
        print(f'[INFO] {clf_name} is trained in %.2f seconds\n' %(runtime_train))

        runtime_train = 0.0
        start_time_train = 0.0

    #Saving results in csv
    df_gridsearch_results_params = pd.DataFrame(gridsearch_results, 
                                                columns=['classifier', 
                                                         'best_score', 
                                                         'best_params', 
                                                         'best_estimator', 
                                                         'time'])
    df_gridsearch_results_params.to_csv(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", 'ML_df_gridsearch_results_params.csv'), index=None)
    df_ml_results = pd.DataFrame(scores_results, 
                                 columns=['classifier', 
                                          'accuracy_score', 
                                          'accuracy_score_std', 
                                          'roc_auc_score', 
                                          'roc_auc_score_std', 
                                          'f1_score', 
                                          'f1_score_std', 
                                          'precision_score', 
                                          'precision_score_std', 
                                          'recall_score', 
                                          'recall_score_std', 
                                          'classification_report', 
                                          'confusion_matrix',
                                          'time'])
    df_ml_results.to_csv(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", 'ML_df_machine_learning_results.csv'), index=None)

    return best_model_voting


def voting_classifier(model_name, num_deep_radiomics, X_train, X_test, y_train, y_test, best_model_voting) -> None:
    """ Voting Classifier using Ensemble Learning Model

    Args:
        model_name (_type_): _description_
        num_deep_radiomics (_type_): _description_
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        best_model_voting (_type_): _description_
    """   

    runtime_train = 0.0
    start_time_train = 0.0
    scores_results_voting = []

    voting_classifier = VotingClassifier(estimators=best_model_voting, voting='hard')

    print(f'[INFO] Voting Classifier using Ensemble Learning Model is trainning...')
    start_time_train = time.time()
    voting_classifier.fit(X_train, y_train)
    final_preds = voting_classifier.predict(X_test)

    accuracy_score_ = accuracy_score(y_test, final_preds)
    roc_auc_score_ = roc_auc_score(y_test, final_preds)
    f1_score_ = f1_score(y_test, final_preds, average='macro')
    precision_score_ = precision_score(y_test, final_preds, average='macro')
    recall_score_ = recall_score(y_test, final_preds, average='macro')
    classification_report_ = classification_report(y_test, final_preds)
    confusion_matrix_ = confusion_matrix(y_test, final_preds)

    runtime_train = time.time() - start_time_train

    scores_results_voting.append({
        'classifier': voting_classifier.__class__.__name__,
        'accuracy_score': accuracy_score_,
        'roc_auc_score': roc_auc_score_,
        'f1_score': f1_score_,
        'precision_score': precision_score_,
        'recall_score': recall_score_,
        'classification_report': classification_report_,
        'confusion_matrix': confusion_matrix_,
        'time': runtime_train
    })
        
    print(f"Acurracy: {accuracy_score_}")
    print(f"ROC AUC: {roc_auc_score_}")
    print(f"F1-Score: {f1_score_}")
    print(f"Precision: {precision_score_}")
    print(f"Recall: {recall_score_}")
    print(f"Classification Report: \n {classification_report_}")
    print(f"Confusion Matrix: \n {confusion_matrix_}")
    print(f'[INFO] Voting Classifier using Ensemble Learning Model is trained in %.2f seconds\n' %(runtime_train))

    df_ensemble_learning_results = pd.DataFrame(scores_results_voting,\
                                                columns=['classifier', 
                                                         'accuracy_score', 
                                                         'roc_auc_score', 
                                                         'f1_score', 
                                                         'precision_score', 
                                                         'recall_score', 
                                                         'classification_report', 
                                                         'confusion_matrix',
                                                         'time']) 
    df_ensemble_learning_results.to_csv(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", 'ML_df_ensemble_learning_results.csv'), index=None)
