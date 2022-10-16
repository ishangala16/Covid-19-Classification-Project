import streamlit as st
import pickle
import os 
import pandas as pd
import numpy as np
import cv2
from tensorflow import keras

def prediction(img):
    model = keras.models.load_model('cnn_model.h5')
    predicition = model.predict(img)
    predicition = np.argmax(predicition, axis = 1) 
    if predicition == 1:
        result = 'Positive'
    else:
        result = 'Negative'
    st.subheader('Prediction')
    st.markdown(f'The patient is Covid : **{result}**') 

def get_list_of_images():
    file_list = os.listdir('test_images')
    return [str(filename) for filename in file_list if str(filename).endswith('.png')]

def load_image():
    uploaded_file = st.sidebar.file_uploader(label='Pick an image to test')
    if uploaded_file and st.sidebar.button('Load'):
        image = keras.preprocessing.image.load_img(uploaded_file)
        # print(type(image))
        with st.expander('Selected Image', expanded = True):
            st.image(image, use_column_width = True)
        image = keras.preprocessing.image.load_img(uploaded_file, target_size=(70,70))
        image = keras.preprocessing.image.img_to_array(image)
        # img = cv2.resize(image, (70,70))/255
        print('############')
        detection = image/255   
        detection = np.expand_dims(detection, axis=0)

        print(detection.shape)
        prediction(detection)


def main():
    st.title('Covid-19 Classifier')
    st.sidebar.title("Upload an Image")
    load_image()


if __name__ == '__main__':
    main()