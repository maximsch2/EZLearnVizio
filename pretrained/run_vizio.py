from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import sys
import h5py
from PIL import Image






model = ResNet50(weights='imagenet', include_top=False)

with h5py.File("saved_weights.h5", "r") as saved_weights:
    EZLearn_W = np.array(saved_weights["W"])
    EZLearn_b = np.array(saved_weights["b"])
    EZLearn_names = np.array(saved_weights["names"])


def load_img_resized(img_path, maxsize):
    img = image.load_img(img_path)
    s = img.size
    ratio = min(maxsize*1.0/s[0], maxsize*1.0/img.height)
    finalsize = (int(ratio*s[0]), int(ratio*s[1]))    
    img = img.resize(finalsize, Image.BILINEAR)
    size = (maxsize, maxsize)
    background = Image.new('RGB', size, (255, 255, 255))
    background.paste(
        img, (int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2))
    )
    return background

def get_features_for_im(img_path):    
    img = load_img_resized(img_path, 224)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = model.predict(x)
    return features.flatten()


def softmax(X):
    tmp = np.exp(X)
    return tmp/tmp.sum()

def get_predictions_for_im(img_path):
    features = get_features_for_im(img_path)
    predictions = softmax(EZLearn_W.dot(features) + EZLearn_b) 
    for prob, name in zip(predictions, EZLearn_names):
        if prob > 0.3:
            print name, prob


get_predictions_for_im(sys.argv[1])