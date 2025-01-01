#!/bin/bash

#Some models from keras: https://keras.io/api/applications/
list_model=("VGG16" "ResNet50V2" "DenseNet121" "DenseNet201" "InceptionV3" "EfficientNetB3")

#Range for num_deep_radiomics
num_deep_radiomics=("100" "300" "500")

#Number of epochs
epochs=100

#Loop model
for model in "${list_model[@]}"; do
    #Num deep radiomics
    for num in "${num_deep_radiomics[@]}"; do
        #Execute the Python command with the desired arguments
        python3 main.py --model_name "$model" --num_deep_radiomics "$num" --epochs "$epochs"
    done
done
