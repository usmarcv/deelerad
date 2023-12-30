import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input as vp
from keras import backend as K
import argparse
import os
import pathlib
import matplotlib.pyplot as plt


def get_img_array(img_path, size):
    """convert PIL image to numpy array

    Args:
        img_path (str): path image 
        size (int): image size

    Returns:
        np.array: image array
    """
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """make the heatmaps of the gradcam

    Args:
        img_array (np.array): unique image to generate the gradcam
        model (keras model): model pre-treinaed on images
        last_conv_layer_name (str): last convolutional layer of the model
        pred_index (_type_, optional): _description_. Defaults to None.

    Returns:
        np.array: heapmaps generated
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def gradcam_processing(img_path, heatmap, cam_path="cam.jpg", alpha=0.4, color_map="jet"):
    """processing grad-cam image

    Args:
        img_path (str): image path
        heatmap (np.array): heatmap array
        cam_path (str, optional): name to save the heatmap . Defaults to "cam.jpg".
        alpha (float, optional): _description_. Defaults to 0.4.
        color_map (str, optional): coloramp ('jet', 'reds', 'RdGy_r'). Defaults to 'jet'

    Returns:
        _type_: _description_
    """
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    colormap = cm.get_cmap(color_map)

    # Use RGB values of the colormap
    colormap_colors = colormap(np.arange(256))[:, :3]
    colormap_heatmap = colormap_colors[heatmap]

    # Create an image with RGB colorized heatmap
    colormap_heatmap = tf.keras.preprocessing.image.array_to_img(colormap_heatmap)
    colormap_heatmap = colormap_heatmap.resize((img.shape[1], img.shape[0]))
    colormap_heatmap = tf.keras.preprocessing.image.img_to_array(colormap_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = colormap_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    #tf.keras.preprocessing.image.save_img(cam_path, superimposed_img)
    return superimposed_img
    #superimposed_img.save(cam_path)
    #plt.savefig(superimposed_img, cam_path, bbox_inches='tight')
    #return superimposed_img
    
    # Save the superimposed image
    #superimposed_img.save(cam_path)
    
def get_grad_cam(img_path, model, last_conv_layer_name, image_size, to_path, color_map="jet"):
    """get and save grad-cam image

    Args:
        img_path (str): image path
        model (keras model): model 
        last_conv_layer_name (str): last convolutional layer of the model
        image_size (int): image size
        to_path (str): path to save gradcam image
        color_map (str, optional): coloramp ('jet', 'reds', 'RdGy_r'). Defaults to 'jet'
    """
    model.layers[-1].activation = None 
    img_path = str(img_path)
   

    img_array = vp(get_img_array(img_path, size=image_size))
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    grad  = gradcam_processing(img_path, heatmap, color_map=color_map)
    predict = np.argmax(model.predict(img_array), axis=1)[0]
    print("Image path: {} -- pred value: {}".format(img_path, label[predict]))
    
    #if not os.path.exists(os.path.join(to_path, str(img_path).split(BARVALUE)[-2])):
    os.makedirs(os.path.join(to_path, str(img_path).split(BARVALUE)[-2], color_map), exist_ok=True)

    tf.keras.preprocessing.image.save_img(os.path.join(to_path, str(img_path).split(BARVALUE)[-2], color_map, str(img_path).split(BARVALUE)[-1]), grad)
    #tf.keras.preprocessing.image.save_img(os.path.join(to_path, str(img_path).split("/")[-2], "heatmap_{}".format(str(img_path).split("/")[-1])), heatmap)
    #cv2.imwrite(os.path.join(to_path, str(img_path).split("/")[-2], "heatmap_{}".format(str(img_path).split("/")[-1])), heatmap)
    plt.matshow(heatmap)
    plt.axis('off')
    plt.savefig(os.path.join(to_path, str(img_path).split(BARVALUE)[-2], color_map, "heatmap_{}".format(str(img_path).split(BARVALUE)[-1])), bbox_inches='tight', pad_inches=0)
    

if __name__ == "__main__":

    BARVALUE = "\\" if os.name == "nt" else "/"
    label = {0:"COVID", 1: "NORMAL"}
    parser = argparse.ArgumentParser()
    # Add the path that have the images to generate the grad-cam
    parser.add_argument('--image_from', type=str, help="image path to generate the gradcam",required=True)
    parser.add_argument('--model_weights', help="weights .hdf5 generated on gradcam",type=str, required=True)
    parser.add_argument('--model_name', type=str, help="model name", required=True)
    parser.add_argument('--to_path', type=str, help="path to save gradcam images", required=True)
    #parser.add_argument('--original', type=bool, required=False, action="store_true")

    # Parse the argument
    args = parser.parse_args()

    #imgs_path_root = "attacks_images"
    imgs_path_root = args.image_from
    image_size = (224, 224)
    #files = glob.glob(os.path.join(glob.escape(p) + "/*.jpeg"), recursive=True)
    #img = os.path.join("attacks_images", "Deep_resnet50_chest_xray_attack", "0.1", "PNEUMONIA", "person51_virus_105.jpeg")
    #last_conv_layer_name = "conv5_block3_out" # Block Resnet50
    last_conv_layer_name = {
        "resnet50": "conv5_block3_out", 
        "inceptionv3": "conv2d_93",
        "vgg16": "block5_conv3"
    }
    model_name = args.model_weights
    cnn = args.model_name
    to_path = args.to_path

    model_trained = load_model(os.path.join(model_name))
        
    p_from = os.path.join(imgs_path_root)
                    
    files = list(pathlib.Path(p_from).rglob("*"))
                    
    os.makedirs(to_path, exist_ok=True)
                    
    for color_map in ['jet', 'Reds', 'RdGy_r']:
        [get_grad_cam(f, model_trained, last_conv_layer_name[cnn], image_size, to_path, color_map=color_map) for f in files]
            
            