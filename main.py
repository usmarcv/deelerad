"""Import all the necessary packages"""
import pandas as pd
import random
import os
import cv2
import csv
import copy
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import hiss.baseline_test.dl_reform.utils as utils
import deep_learning_models
import machine_learning_models

""""""

RANDOM_SED: int = 42

np.random.seed(RANDOM_SED)
tf.random.set_seed(RANDOM_SED)
keras.utils.set_random_seed(RANDOM_SED)


"""Test for GPU or CPU devices"""
print("Device: \n", tf.config.experimental.list_physical_devices())
print(tf.__version__)
print(tf.test.is_built_with_cuda())


parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model_name', type=str, default="VGG16", required=True, help="CNN name such as: VGG16, VGG-19, ResNet50V2, DenseNet121, DenseNet201, InceptionV3 or EfficientNetB3")
parser.add_argument('-ndr', '--num_deep_radiomics', type=str, required=True, help="Number of deep radiomics: 100, 200, 300, 400")
args = vars(parser.parse_args())


if __name__ == '__main__':

    if not os.path.exists(os.path.join('./', args['model_name'])):
        os.mkdir(os.path.join('./', args['model_name']))
    
    if not os.path.exists(os.path.join('./', args['model_name'], f"_feat_{args['num_deep_radiomics']}" )):
        os.mkdir(os.path.join('./', args['model_name'], f"_feat_{args['num_deep_radiomics']}"))


    #Path directory
    covid = r'/home/marcuslobo/codes/hiss/baseline_test/dl_reform/dataset_script/covid/'   #directory of covid image
    non_covid = r'/home/marcuslobo/codes/hiss/baseline_test/dl_reform/dataset_script/normal/'   #directory of non-covid image

    images_covid = utils.load_images_from_folder(covid)
    images_non_covid = utils.load_images_from_folder(non_covid)
    images = images_covid + images_non_covid

    label_dl, label_ml, images = utils.preprocessing_dataset(images)

    NUM_CLASSES: int = 2
    input_shape = (224, 224, 3)


    #Split dataset to DL models
    (X_train, X_test, y_train, y_test) = train_test_split(images, label_dl, test_size=0.20, stratify=label_dl, random_state=RANDOM_SED)

    model = deep_learning_models.create_model(args['model_name'], input_shape, int(args['num_deep_radiomics']), NUM_CLASSES)

    training = deep_learning_models.training_model(model, args['model_name'], int(args['num_deep_radiomics']), X_train, y_train, X_test, y_test)

    save_features = deep_learning_models.save_deep_radiomic_features(args['model_name'], int(args['num_deep_radiomics']), images, label_ml)

    X_train_rad, X_test_rad, y_train_rad, y_test_rad = utils.preprocessing_radiomic_features(args['model_name'], int(args['num_deep_radiomics']))

    best_model_gs = machine_learning_models.training_gridsearchcv_best_estimator(args['model_name'], int(args['num_deep_radiomics']), X_train_rad, X_test_rad, y_train_rad, y_test_rad)

    machine_learning_models.voting_classifier(X_train_rad, X_test_rad, y_train_rad, y_test_rad, best_model_gs)
    
    
