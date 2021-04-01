import streamlit as st
import cv2
from keras.models import load_model,Model
import numpy as np
from PIL import Image
model=load_model("Dogs-vs-Cats_model.h5")

def config_image(img):
    images=[]
    preprocess_images=[]
    if img is not None:
        rescale=cv2.resize(img,(256,256))/255.0
        images.append(img)
        preprocess_images.append(rescale)
    return images,preprocess_images
# """
#         if pred>0.5, model classifi image inclue dog
#         else is cat
# """
st.write("\n")


st.sidebar.title("upload inmage")
st.set_option('deprecation.showfileUploaderEncoding',False)
upload_file=st.sidebar.file_uploader(" ",type=["jpg","png","jpeg"])

if st.sidebar.button("click here to classifi"):
    if upload_file is None:

        st.sidebar.write("please upload file before")
    else:
        img = Image.open(upload_file)
        st.image(img, use_column_width=False, width=300)
        img = np.array(img)

        imgs, preprocess = config_image(img)
        predict = model.predict(np.asarray(preprocess))


        if predict>0.5:
            st.write("Dog")
        else:
            st.text("cat")

