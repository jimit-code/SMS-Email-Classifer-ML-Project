import os
os.environ['NLTK_DATA'] = os.path.join(os.path.dirname(__file__), 'nltk_data')

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('nb_model.pkl', 'rb') as f:
    model = pickle.load(f)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def processor(text):

    #first time to lower case the text
    text = text.lower()

    #remove any kind of extra symbols
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    #tokenize the word
    words = word_tokenize(text)

    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)



st.title('Email/SMS Spam Classifer')
user_input = st.text_area('Enter your SMS text')

if st.button('Classify'):
    clean_msg = processor(user_input)
    vect = tfidf.transform([clean_msg])
    prediction = model.predict(vect)
    result = 'Spam' if prediction[0] == 1 else "Not Spam"
    st.subheader(f'Prediction: {result}')