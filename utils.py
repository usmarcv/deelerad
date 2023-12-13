import os
import cv2
import copy
import random
import numpy as np
import pandas as pd

#Sklearn and Tensorflow libraries
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_SED: int = 42

np.random.seed(RANDOM_SED)


def clahe_function(img) -> np.ndarray:
    """ Function for applying CLAHE (Contrast Limited Adaptive Histogram Equalization) to images

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """ 

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe_image = clahe.apply(l)
    merge_image_lab = cv2.merge((clahe_image, a, b))
    image_clahe_transformed = cv2.cvtColor(merge_image_lab, cv2.COLOR_LAB2BGR)
    
    return image_clahe_transformed


def load_images_from_folder(folder) -> list:
    """ Function for loading images from folder

    Args:
        folder (_type_): _description_

    Returns:
        _type_: _description_
    """    

    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = clahe_function(img)
        img = img/255.0
        img = cv2.resize(img, (224, 224))

        if img is not None:
            images.append(img)

    return images


def preprocessing_dataset(dataset_images) -> tuple:
    """ Preprocessing dataset with labels for DL and ML models

    Args:
        dataset_images (_type_): _description_

    Returns:
        _type_: _description_
    """    

    #Labeling the dataset
    #Set 1 to normal/healthy images and 0 to covid images
    y = np.ones(196) 
    y = np.append(y, np.zeros(196))  
    y = list(y) #list to labels from load images
    c = list(zip(dataset_images, y)) #list all instances

    #Shuffling the dataset
    random.shuffle(c)
    dataset_images, y = zip(*c)
    del c  #free memory

    #Converting to numpy array
    dataset_images = np.array(dataset_images)
    y = np.array(y) #label to DL models
    label_ml = copy.deepcopy(y) #label to ML models

    #Labeling for DL model 
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    label_dl = to_categorical(y)

    return label_dl, label_ml, dataset_images


def preprocessing_radiomic_features(model_name, num_deep_radiomics) -> tuple:
    """ Preprocessing radiomic features for ML models

    Args:
        model_name (_type_): _description_
        num_deep_radiomics (_type_): _description_

    Returns:
        _type_: _description_
    """   

    #Loading the dataset
    dataset = pd.read_csv(os.path.join('./', model_name, f"{num_deep_radiomics}_deepradiomics", 
                                       f'DL_{model_name}_{num_deep_radiomics}_extracted_deepradiomics_withlabel.csv'))

    #Splitting data into csv
    y = dataset.iloc[:,-1:]
    dataset = dataset.loc[:, dataset.columns]
    dataset = dataset.iloc[:,:-1]

    #Normalizing the data
    x = dataset.loc[:, dataset.columns].values
    x = StandardScaler().fit_transform(x)

    #Splitting the data into train and test
    (X_train, X_test, y_train, y_test) = train_test_split(x, y.values.ravel(), test_size=0.20, stratify=y, random_state=RANDOM_SED)
    print('\n[INFO] Dataset using deep radiomics features')
    print('\tTrain set: ', X_train.shape)
    print('\tTrain label set: ', y_train.shape)
    print('\tTest set: ', X_test.shape)
    print('\tTest label set: \n', y_test.shape)

    return X_train, X_test, y_train, y_test
