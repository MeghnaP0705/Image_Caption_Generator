import gradio as gr
import json
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

# Load the word-to-index and index-to-word mappings
with open("word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file)

with open("idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file)

# Load the model
model = load_model('model_checkpoints/model_19.h5')

# Load ResNet50 model for feature extraction
resnet50_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
resnet50_model = Model(resnet50_model.input, resnet50_model.layers[-2].output)

# Function to generate captions
def predict_caption(photo):
    inp_text = "startseq"
    for i in range(38):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=38, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]

        inp_text += (' ' + word)
        if word == 'endseq':
            break
    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

# Preprocess image for ResNet50
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Extract image features
def encode_image(img):
    img = preprocess_image(img)
    feature_vector = resnet50_model.predict(img)
    return feature_vector
# Function for Gradio Interface
def generate_caption(img):
    photo = encode_image(img).reshape((1, 2048))
    caption = predict_caption(photo)
    return caption
# Gradio UI
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Image Caption Generator",
    description="Upload an image and get a generated caption based on the trained model."
)
# Launch the Gradio app
iface.launch()
