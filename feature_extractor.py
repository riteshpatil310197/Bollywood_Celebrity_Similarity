import os
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm

import pickle

filenames = pickle.load(open('filenames.pkl','rb'))
#print(filenames)

model = VGGFace(model = 'resnet50',include_top=False, input_shape=(224,224,3),pooling='avg')

#print(model.summary())

def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    #print(img_array.shape)
    expanded_img = np.expand_dims(img_array,axis=0)
    #print(expanded_img.shape)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(expanded_img).flatten()
    return result

features= []
for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embeddings.pkl','wb'))



'''actors = os.listdir('Data')

fileNames = []
for actor in actors:
    for file in os.listdir(os.path.join('Data',actor)):
        fileNames.append(os.path.join('Data',actor,file))

#print(len(fileNames))

pickle.dump(fileNames,open('filenames.pkl','wb'))'''




