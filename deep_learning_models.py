import os
import csv
import time
import numpy as np
import pandas as pd

#Deep Learning Framework
import tensorflow as tf

#Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class batch_timer(tf.keras.callbacks.Callback):
    """ Callback to save the time of training and testing

    Args:
        tf (_type_): _description_
    """    
    def __init__(self, file_train, file_test):
        super(batch_timer, self).__init__()
        self.file_test = file_test
        self.file_train = file_train
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time_train = time.time()    
    
    def on_epoch_end(self, epoch, logs=None):
        stop_time_train = time.time()
        time_train = stop_time_train-self.start_time_train
        with open(file=self.file_train, mode="a", encoding="UTF8", newline="") as csvW:
            writer = csv.writer(csvW)
            writer.writerow([str(epoch), str(time_train)])
    
    def on_test_begin(self, logs=None):
        self.start_time_test = time.time()

    def on_test_end(self, logs=None):
        stop_time_test = time.time()
        time_train = stop_time_test-self.start_time_test
        with open(file=self.file_test, mode="a", encoding="UTF8", newline="") as csvW:
            writer = csv.writer(csvW)
            writer.writerow([str(time_train)])


#Metrics for Deep Learning models
METRICS = [
    "accuracy",
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def create_model(model_name, num_deep_radiomics, input_shape, num_classes)-> object:
    """ Create a Deep Learning model with transfer learning and fine-tuning

    Args:
        model_name (_type_): _description_
        num_deep_radiomics (_type_): _description_
        input_shape (_type_): _description_
        num_classes (_type_): _description_

    Returns:
        object: _description_
    """    

    base_model = eval("tf.keras.applications." + model_name + "(weights = 'imagenet', input_shape=input_shape, include_top = False)")    

    for layer in base_model.layers:
        layer.trainable = False                                              
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(num_deep_radiomics, activation='relu', name='layer_deep_radiomics')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    preds = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs = base_model.input, outputs = preds)

    for layer in model.layers:
        if layer.trainable != False:
            layer.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=METRICS)

    return model


def training_model(model, model_name, num_deep_radiomics, input_shape, X_train, y_train, X_test, y_test, epochs)-> None:
    """ Training a Deep Learning model with transfer learning and fine-tuning

    Args:
        model (_type_): _description_
        model_name (_type_): _description_
        num_deep_radiomics (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
    """    

    lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=1e-5, patience=5, verbose=0)
    early       = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
    filepath    = os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", f'DL_{model_name}_{num_deep_radiomics}_best_model.keras' )
    checkpoint  = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    scores_results_dl = []
    runtime_train = 0.0
    start_time_train = 0.0
    
    with tf.device('/GPU:0'):
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", f'DL_{model_name}_{num_deep_radiomics}_metrics.csv'))
        start_time_train = time.time()
        history = {}
        history = model.fit(X_train, 
                            y_train, 
                            batch_size=32, 
                            epochs=epochs, 
                            verbose=1, 
                            validation_data=(X_test, y_test), 
                            callbacks=[early, 
                                       lr_reduce, 
                                       csv_logger, 
                                       checkpoint, 
                                       batch_timer(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", f'DL_{model_name}_{num_deep_radiomics}_time_train.csv'),
                                                   os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", f'DL_{model_name}_{num_deep_radiomics}_time_test.csv'))
                                        
                                        ]
                            )
    
    #Save trained model
    model.save(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", f'DL_{model_name}_{num_deep_radiomics}_trained_model.keras'))
    runtime_train = time.time() - start_time_train

    y_test_arg = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(f'\n[INFO] Model {model_name} performance...\n')

    #Metrics
    accuracy_score_ = accuracy_score(y_test_arg, y_pred)
    roc_auc_score_ = roc_auc_score(y_test_arg, y_pred)
    f1_score_ = f1_score(y_test_arg, y_pred, average='macro')
    precision_score_ = precision_score(y_test_arg, y_pred, average='macro')
    recall_score_ = recall_score(y_test_arg, y_pred, average='macro')
    classification_report_ = classification_report(y_test_arg, y_pred)
    confusion_matrix_ = confusion_matrix(y_test_arg, y_pred)

    print(f"Acurracy: {accuracy_score_}")
    print(f"ROC AUC: {roc_auc_score_}")
    print(f"F1-Score: {f1_score_}")
    print(f"Precision: {precision_score_}")
    print(f"Recall: {recall_score_}")
    print(f"Classification Report: \n {classification_report_}")
    print(f"Confusion Matrix: \n {confusion_matrix_}")

    scores_results_dl.append({
        'classifier': model_name,
        'accuracy_score': accuracy_score_,
        'roc_auc_score': roc_auc_score_,
        'f1_score': f1_score_,
        'precision_score': precision_score_,
        'recall_score': recall_score_,
        'classification_report': classification_report_,
        'confusion_matrix': confusion_matrix_,
        'time': runtime_train
    })
        
    df_dl_results = pd.DataFrame(scores_results_dl,
                                 columns=['classifier', 
                                          'accuracy_score', 
                                          'roc_auc_score', 
                                          'f1_score', 
                                          'precision_score', 
                                          'recall_score', 
                                          'classification_report', 
                                          'confusion_matrix',
                                          'time'])
    df_dl_results.to_csv(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", 
                                      f'DL_{model_name}_{num_deep_radiomics}_sklearn_results.csv'), index=None)


def save_deep_radiomic_features(model_name, num_deep_radiomics, dataset_images, label_ml) -> None:
    """ Save deep radiomic features extracted from Deep Learning models

    Args:
        model_name (_type_): _description_
        num_deep_radiomics (_type_): _description_
        dataset_images (_type_): _description_
        label_ml (_type_): _description_
    """    

    model = tf.keras.models.load_model(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", 
                                                    f'DL_{model_name}_{num_deep_radiomics}_trained_model.keras'))
    layer_deep_radiomics = tf.keras.Model(inputs=model.input, outputs=model.get_layer('layer_deep_radiomics').output)

    deep_radiomics_features_extracted = layer_deep_radiomics.predict(dataset_images)
    deep_radiomics_features_extracted = pd.DataFrame(deep_radiomics_features_extracted)
    deep_radiomics_features_extracted.to_pickle(os.path.join('./', model_name, 
                                                             f"{num_deep_radiomics}_deepradiomics", 
                                                             f'DL_{model_name}_{num_deep_radiomics}_extracted_deepradiomics_notlabel.pkl'))
    
    deep_radiomics_features = pd.read_pickle(os.path.join('./', model_name, 
                                                          f"{num_deep_radiomics}_deepradiomics", 
                                                          f'DL_{model_name}_{num_deep_radiomics}_extracted_deepradiomics_notlabel.pkl'))

    deep_radiomics_features['label'] = label_ml                         
    deep_radiomics_features.to_csv(os.path.join('./', model_name, 
                                                f"{num_deep_radiomics}_deepradiomics", 
                                                f'DL_{model_name}_{num_deep_radiomics}_extracted_deepradiomics_withlabel.csv'), index=None)