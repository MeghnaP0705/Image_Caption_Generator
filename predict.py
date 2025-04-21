import json
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import collections
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Ensure nltk package is installed: pip install nltk

# Load word-to-index and index-to-word mappings
with open("word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file)

with open("idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file)

# Load the trained model
print("Loading the model...")
model = load_model('model_checkpoints/model_19.h5')

# Load test image encodings
with open("encoded_test_features.pkl", "rb") as file:
    test_encoding = pd.read_pickle(file)

# Function to parse captions.txt and create a dictionary
def load_captions(file_path):
    captions_dict = collections.defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")  # Split image ID and caption
            if len(parts) == 2:
                img_id_full, caption = parts
                img_id = img_id_full.split("#")[0].replace(".jpg", "")  # Remove ".jpg" to match test_encoding keys
                captions_dict[img_id].append(caption)
    return captions_dict

# Load ground truth captions from captions.txt
captions_dict = load_captions("data/textFiles/tokens.txt")

# Function to predict caption
def predict_caption(photo):
    inp_text = "startseq"
    for i in range(38):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=38, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word.get(ypred, "")

        inp_text += (' ' + word)
        if word == 'endseq':
            break

    final_caption = inp_text.split()[1:-1]  # Remove startseq and endseq
    return ' '.join(final_caption)

# Get a random image
all_img_IDs = list(test_encoding.keys())
number = np.random.randint(0, len(test_encoding))
img_ID = all_img_IDs[int(number)]
photo = test_encoding[img_ID].reshape((1, 2048))

print(f"Selected Image ID: {img_ID}")
print("Generating caption...")
predicted_caption = predict_caption(photo)

# Display the image
img_path = f"data/Images/{img_ID}.jpg"
img_data = plt.imread(img_path)
plt.imshow(img_data)
plt.axis("off")
plt.show()

# Compute BLEU score
if img_ID in captions_dict:
    reference_captions = [cap.split() for cap in captions_dict[img_ID]]  # Tokenize reference captions
    predicted_tokens = predicted_caption.split()
    bleu_score = sentence_bleu(reference_captions, predicted_tokens)
    print("Generated Caption:", predicted_caption)
    print("BLEU Score:", bleu_score)
else:
    print(f"No reference captions available for image: {img_ID}")
