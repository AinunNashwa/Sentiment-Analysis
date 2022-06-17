# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:52:09 2022

@author: User
"""

#%% Deployment unsually done on another PC/mobile phone
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import os
import json
import pickle
import numpy as np
import re

#1) Trained model --> loading from h.5
#2) Tokenizer --> loading from json
#3) MMS/OHE --> loading from pickle

# to load trained model
loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))

loaded_model.summary()

# to load tokenizer
TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_sentiment.json')
with open(TOKENIZER_PATH, 'r') as json_file:
    loaded_tokenizer = json.load(json_file)
    
# to load pickle
OHE_PICKLE_PATH = os.path.join(os.getcwd(), 'ohe.pkl')
with open (OHE_PICKLE_PATH, 'rb') as file:
    loaded_ohe = pickle.load(file)

#%%

tokenizer = tokenizer_from_json(loaded_tokenizer)

input_review = input('Type your review here: ')

input_review = re.sub('< *?>',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ',input_review).lower().split()

input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = pad_sequences(np.array(input_review_encoded).T,maxlen=180,
                                     padding='post',
                                     truncating='post')

outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

print(loaded_ohe.inverse_transform(outcome))














