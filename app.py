import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity

detector = MTCNN()
model = VGGFace(model = 'resnet50',include_top=False, input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('embeddings.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))



st.title("Which Bollywood Celebrity are you !")

def save_uploaded_Image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False
    
def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x,y,width,height = results[0]['box']
    face = img[y:y+height,x:x+width]

#Extract Features

    image=Image.fromarray(face)
    image=image.resize((224,224))


    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img=np.expand_dims(face_array,axis=0)

    preprocessed_img = preprocess_input(expanded_img)

    result= model.predict(preprocessed_img).flatten()
    return result
uploaded_image = st.file_uploader("Choose an image")

def recommend(feature_list,feature):
    similarity=[]
    #Find cosine similarity
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])
    index_position = sorted(list(enumerate(similarity)),reverse=True,key=lambda x: x[1])[0][0]
    return index_position

if uploaded_image is not None:
    if save_uploaded_Image(uploaded_image):
        display_Image = Image.open(uploaded_image)
        #extract features
        features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)
        #recommend
        index_position = recommend(feature_list,features)
        #display
        col1,col2=st.beta_columns(2)
        with col1:
            st.header("Your Uploaded Image")
            st.image(display_Image)
        with col2:
            st.header(" ".join(filenames[index_position].split('\\')[1].split('_')))
            st.image(filenames[index_position])


        
