import os
import cv2
import copy
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clahe_function(img):
  
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) Method for preprocessing all images

    params

    
    """  
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    clahe_img = clahe.apply(l)
    updated_lab_img2 = cv2.merge((clahe_img, a, b))
    # Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    
    return CLAHE_img


"""Funcntion for loading images from dataset"""

def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #img = cv2.resize(img, (224, 224))
        img = clahe_function(img)
        img = img/255.0
        img = cv2.resize(img, (224, 224))
        if img is not None:
            images.append(img)

    return images


def preprocessing_dataset(dataset_images):

    y = np.ones(196)  #labeling healthty/normal images as 1
    y = np.append(y, np.zeros(196))  #labeling covid images as 0
    y = list(y) #list to labels from load images
    c = list(zip(dataset_images, y)) #list all instances

    #reshuffling all the images along with their labels
    random.shuffle(c)
    dataset_images, y = zip(*c)
    del c  #For Memory Efficiency

    dataset_images = np.array(dataset_images)
    y = np.array(y) #label to DL models
    label_ml = copy.deepcopy(y) #label to ML models

    #Labeling for DL model 
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    label_dl = to_categorical(y)

    return label_dl, label_ml, dataset_images


def preprocessing_radiomic_features(model_name, num_deep_radiomics):

    dataset = pd.read_csv(os.path.join('./', model_name, f"_feat_{num_deep_radiomics}", '_final_deepradiomics_withlabel.csv'))

    #Splitting data into csv
    y = dataset.iloc[:,-1:]
    dataset = dataset.loc[:, dataset.columns]
    dataset = dataset.iloc[:,:-1]

    """Normalization of The Features"""

    x = dataset.loc[:, dataset.columns].values
    x = StandardScaler().fit_transform(x)
    #y = StandardScaler().fit_transform(y)


    """Splitting the Data into Training & Testing Set"""
    (X_train, X_test, y_train, y_test) = train_test_split(x, y.values.ravel(), test_size=0.20, stratify=y, random_state=42)

    print('[INFO] Dataset...')
    print('\tTrain set: ', X_train.shape)
    print('\tTrain label set: ', y_train.shape)
    print('\tTest set: ', X_test.shape)
    print('\tTest label set: ', y_test.shape)

    return X_train, X_test, y_train, y_test
