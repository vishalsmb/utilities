import torch
import streamlit as st
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
import numpy as np
from PIL import Image
from variables import device, model, log_format, pinecone_index
from general_utils import preprocess, download_mnist_dataset
from pinecone_utils import convert_to_embedding, query_index
import logging
from sys import platform
import os

st.set_page_config(layout='wide')
if not os.path.isdir(os.path.join("mnist_images", "train")):
    download_mnist_dataset()

# Setting the title
st.title("MNIST Image Similarity Search")
# Image Upload
uploaded_image = st.file_uploader("Upload your Image", type=['png', 'jpg'])

if uploaded_image is not None:
    st.image(Image.open(uploaded_image), caption='Uploaded Image', width=200)
    query_embeddings = convert_to_embedding(model.encoder, [uploaded_image])
    top_k = 5
    query_response = query_index(pinecone_index, query_embeddings[0], top_k)

    similar_images = []
    for i in range(0, top_k):
        image_path = query_response["matches"][i]['metadata']['file_path']
        if platform != "win32":
            image_path = image_path.replace("\\", '/')

        loaded_image = np.array(Image.open(image_path))
        similar_images.append(loaded_image)

    st.image(similar_images, width=200)
