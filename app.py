import streamlit as st
import re
import pickle
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') 
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
porter = PorterStemmer()


def remove_stop_words(sentence): 
  # Split the sentence into individual words 
  words = sentence.split() 
  filtered_words = []
  # Use a list comprehension to remove stop words 
  filtered_words = [word for word in words if word not in stop_words] 
  
  # Join the filtered words back into a sentence 
  return ' '.join(filtered_words)

def remove_html(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)
def remove_punct(data):
    return data.translate(str.maketrans('','',string.punctuation))
def remove_url(data):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',data)
def stem_sentence(sentence):
    tokens = word_tokenize(sentence)
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def preprocess(data):
    data = data.lower()
    data = remove_html(data)
    data = remove_url(data)
    data = remove_punct(data)
    data = remove_stop_words(data)
    data = stem_sentence(data)
    return data



tfidf = pickle.load(open('vectorizer.pkl','rb'))
model_new = pickle.load(open('model_nb.pkl','rb'))

st.title("GPT CHECKER (beta)")

input_sms = st.text_area("Enter the Text")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = preprocess(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model_new.predict(vector_input)[0]
    
    # 4. Display
    if result == 1:
        st.header("AI Generated")
    else:
        st.header("Human Generated")
        
