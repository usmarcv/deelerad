import numpy as np
import time
import csv
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
from sklearn import preprocessing



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


class batch_timer(tf.keras.callbacks.Callback):
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


#Model fine tuning
def create_model(model_name, input_shape, num_deep_radiomics, num_classes):

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


def training_model(model, model_name, num_deep_radiomics, X_train, y_train, X_test, y_test):

    input_shape = (224, 224, 3)
    lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=1e-5, patience=5, verbose=0)
    early       = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

    filepath    = os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", 'best_model.hdf5' )
    checkpoint  = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')

    #CNN Model Trained
    with tf.device('/GPU:0'):
        history = {}
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", 'DL_metrics.csv'))
        history = model.fit(X_train, 
                            y_train, 
                            batch_size=32, 
                            epochs=100, 
                            verbose=1, 
                            validation_data=(X_test, y_test), 
                            callbacks=[early, 
                                       lr_reduce, 
                                       csv_logger, 
                                       checkpoint, 
                                       batch_timer(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", 'time_train.csv'),
                                                   os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", 'time_test.csv'))
                                        
                                        ]
                            )


    #Save the model 
    model.save(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", 'trained_model.h5'))


    y_test_arg = np.argmax(y_test, axis=1)
    Y_pred = np.argmax(model.predict(X_test), axis=1)


    print(f'\nModel {model_name} Performance\n')
    print('Accuracy: %.3f' % float(accuracy_score(y_test_arg, Y_pred)))
    print('AUC: %.3f' % float(roc_auc_score(y_test_arg, Y_pred)))
    print('F1 score: %.3f' % float(f1_score(y_test_arg, Y_pred, average='macro')))
    print('Recall: %.3f' % float(recall_score(y_test_arg, Y_pred, average='macro')))
    print('Precision: %.3f' % float(precision_score(y_test_arg, Y_pred, average='macro')))

    print('\nClasification report:\n', classification_report(y_test_arg, Y_pred))

    print('Confusion Matrix')
    print(confusion_matrix(y_test_arg, Y_pred))



def save_deep_radiomic_features(model_name, num_deep_radiomics, dataset_images, label_ml):


    model = tf.keras.models.load_model(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", 'trained_model.h5'))
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('layer_deep_radiomics').output)

    feature_engg_data = intermediate_layer_model.predict(dataset_images)
    feature_engg_data = pd.DataFrame(feature_engg_data)

    """Save The Features"""
    #os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", 'final_deepradiomics_notlabel.pkl' )
    feature_engg_data.to_pickle(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", '_final_deepradiomics_notlabel.pkl'))
    deep_radiomics_features = pd.read_pickle(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", '_final_deepradiomics_notlabel.pkl'))

    deep_radiomics_features['label'] = label_ml                         
    deep_radiomics_features.to_csv(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", '_final_deepradiomics_withlabel.csv'), index=None)