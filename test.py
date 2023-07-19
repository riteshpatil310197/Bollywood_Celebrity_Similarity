import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from mtcnn import MTCNN
import cv2
from PIL import Image

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model = 'resnet50',include_top=False, input_shape=(224,224,3),pooling='avg')

detector=MTCNN()

#Face Deteaction
sample_img = cv2.imread('sample/salman_dup.png')
results = detector.detect_faces(sample_img)
x,y,width,height = results[0]['box']
face = sample_img[y:y+height,x:x+width]

#Extract Features

image=Image.fromarray(face)
image=image.resize((224,224))


face_array = np.asarray(image)
face_array = face_array.astype('float32')

expanded_img=np.expand_dims(face_array,axis=0)

preprocessed_img = preprocess_input(expanded_img)

result= model.predict(preprocessed_img).flatten()

#print(result.shape)
similarity=[]
#Find cosine similarity
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

#print(len(similarity))


index_position = sorted(list(enumerate(similarity)),reverse=True,key=lambda x: x[1])[0][0]

temp_img = cv2.imread(filenames[index_position])
cv2.imshow('output',temp_img)
cv2.waitKey(0)

