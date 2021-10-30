import streamlit as st
import pickle
import re
import spacy
from nltk import WordNetLemmatizer

en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
adds = ['subject', 'image']
sw_spacy.update(adds)
def preprocess_text(text):

    # removing email id tags
    text = re.sub('\S*@\S*\s?', ' ', text)

    # Removing url
    text = re.sub(r'http\S+', ' ', text)

    # removing r\n\ pattern
    text = re.sub('[\r\n]+', ' ', text)

    # removing numbers and special characters
    text = re.sub('[^A-Za-z ]+', ' ', text)

    # removing words beginning with capital letters
    text = re.sub('([^.])( [A-Z]\w*)', r'\1', text)

    # Removing words less than length 3
    short = re.compile(r'\W*\b\w{1,3}\b')
    text = short.sub(' ', text)

    # Converting to lowercase
    text = text.lower()

    # Lemmatize
    lm = WordNetLemmatizer()
    text = lm.lemmatize(text)

    # Stopwords removal
    text = ' '.join([word for word in text.split() if word not in sw_spacy])

    # Blank lines
    text = re.sub(r'^$\n', '', text, flags=re.MULTILINE)

    # blank spaces
    text = ' '.join([line for line in text.split('\n') if line.strip() != ''])
    # text=' '.join()
    return text


tf = pickle.load(open('final_vectorizer.pkl', 'rb'))
model = pickle.load(open('final_lsvc_model.pkl', 'rb'))

st.title('Email Classifier')
st.write('This application has been built using Linear SVC model')
input_text = st.text_input('Enter your email')

if st.button('Predict'):
    p_text = preprocess_text(input_text)

    # Vectorize
    vector = tf.transform([p_text])

    # Predict
    result = model.predict(vector)[0]

    # Result
    if result == 0:
        st.header('Abusive')
    else:
        st.header('Non Abusive')

opinion = st.radio('Did you like the app?', ['Yes', 'No'])
if opinion == 'Yes':
    st.write('Thanks for your feedback!!')
else:
    st.write("I will surely work to improve it!!")
