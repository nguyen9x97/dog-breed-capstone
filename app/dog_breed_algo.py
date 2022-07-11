from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from face_detector import face_detector
from tqdm import tqdm
import numpy as np
import json
from keras import backend as K


# Load dog names data
with open('dog_names.json', 'r') as f:
    dog_names = json.loads(f.read())


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def extract_tensor(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def predict_breed(img_path):
    dog_breed_model = load_model(
        '../saved_models/weights.best.InceptionV3.hdf5')

    # extract bottleneck features
    bottleneck_feature = extract_tensor(path_to_tensor(img_path))

    # obtain predicted vector
    predicted_vector = dog_breed_model.predict(bottleneck_feature)
    K.clear_session()

    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def dog_breed_algorithm(img_path):

    if dog_detector(img_path):
        breed_predict = predict_breed(img_path).partition('.')[-1]
        return f'This is probably a dog, and its breed is {breed_predict}'
    elif face_detector(img_path):
        breed_predict = predict_breed(img_path).partition('.')[-1]
        return f'This is probably a human, who looks like an {breed_predict}'
    else:
        return 'Could not detect a human or a dog'
