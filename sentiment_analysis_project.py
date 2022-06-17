# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:16:48 2022

@author: User
"""

import pandas as pd


CSV_URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

# EDA
# Step 1 Data Loading
df = pd.read_csv(CSV_URL)


#%%
df_copy = df.copy() # backup

#%%
# Step 2 Data Inspection

df.head(10)
df.tail(10)
df.info()

df['sentiment'].unique() # to get the unique target
df['review'][0]
df['sentiment'][0]

df.duplicated().sum() # 415 duplicated data
df[df.duplicated()]

# <br /> HTML tags have to be removed
# Numbers canbe filtered

# Step 3 Data Cleaning

# to remove duplicated data
df = df.drop_duplicates()

# remove HTML tags
#'<br /> '
#df['review'][0].replce('<br />', ' ')

review = df['review'].values # features: X
sentiment = df['sentiment'].values # sentiment:y

import re
for index,rev in enumerate(review):
    # remove html tags
    # ? dont be greedy
    # * zero or more occurences
    # any character except new line (/n)
    
    review[index] = re.sub('< *?>',' ',rev)

    # convert into lower case
    # remove numbers
    # ^ means NOT
    review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()
    
# Step 4 Features Selection
# Nothing to select

#%%
# Step 5 Preprocessing

# 1: Convert into lower case -> done in data cleaning
# 2: Tokenization
            
vocab_size = 10000
oov_token = 'OOV'

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review) # to learn all of the words
word_index = tokenizer.word_index
#print(word_index)

train_sequences = tokenizer.texts_to_sequences(review)# to convert into numbers

# 3: Padding & truncating
import numpy as np 

length_of_review = [len(i) for i in train_sequences] # list comprehension
np.median(length_of_review) # to get the number of max length for padding
# can use np.mean/np.median for max_len

max_len = 180

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_review = pad_sequences(train_sequences,
                              maxlen=max_len,
                              padding='post',
                              truncating='post')

# 4: One Hot Encoding for the target

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

import pickle
import os
OHE_PICKLE_PATH = os.path.join(os.getcwd(), 'ohe_sentiment.pkl')
with open (OHE_PICKLE_PATH, 'wb') as file:
    pickle.dump(ohe,file)

# 5: Train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 sentiment,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model Development
# use input, LSTM layers,dropout,dense
# achieve > 90% f1 score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras import Input

from tensorflow.keras.layers import Bidirectional,Embedding

embedding_dim = 64
model = Sequential()
model.add(Input(shape=(180))) # (180,1) # np.shape(X_train)[1:]
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(128,return_sequences=(True))))
#model.add(LSTM(128,return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True,show_layer_names=(True))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

hist = model.fit(X_train,y_train,
                 batch_size=128,
                 validation_data=(X_test,y_test),
                 epochs=3)


#%% model evaluation

import matplotlib.pyplot as plt
hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training loss','validation loss'])
plt.show()

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training accuracy','validation accuracy'])
plt.show()

#%%
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score

results = model.evaluate(X_test,y_test)
print(results)

y_true = y_test
y_pred = model.predict(X_test)

y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(y_pred,axis=1)

#%%
cr = classification_report(y_true,y_pred)
acc_score = accuracy_score(y_true,y_pred)
cm = confusion_matrix(y_true,y_pred)

print(cr)
print(acc_score)
print(cm)

#%% Model Saving
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model_sentiment.h5')
model.save(MODEL_SAVE_PATH)

import json
token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open(TOKENIZER_PATH, 'w') as file:
    json.dump(token_json,file)
    
#%% Discussion/Reporting

# Talk about the results
# Model achieved around 84% accuracy during training
# Recall and f1 score reports 87% and 84% respectively
# However the model starts to overfit after 2nd epoch
# Early stopping can be introduced in future to prevent overfitting
# Increase dropout rate to control overfitting
# Trying with different architecture for example BERT model, transformer
# model, GPT3 may help to improve the model

# 1) results --> discussion on the results
# 2) give suggestion --> how to improve the model
# 3) gather evidences showing what went wrong during training/model development









