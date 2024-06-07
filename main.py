import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re
import string

st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Light gray background */
        margin: 0; /* Remove default margin for body */
        padding: 0; /* Remove default padding for body */
    }
    .st-bw {
        background-color: #eeeeee; /* White background for widgets */
    }
    .st-cq {
        background-color: #cccccc; /* Gray background for chat input */
        border-radius: 10px; /* Add rounded corners */
        padding: 8px 12px; /* Add padding for input text */
        color: black; /* Set text color */
    }

    .st-cx {
        background-color: white; /* White background for chat messages */
    }
    .sidebar .block-container {
        background-color: #f0f0f0; /* Light gray background for sidebar */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add some padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True
)

with open("./LogicRegression_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

def clean_text(input_text):
    input_text = input_text.lower()
    input_text = re.sub('[%s]' % re.escape(string.punctuation), ' ', input_text)
    text_tokens = word_tokenize(input_text)
    input_text = " ".join([word for word in text_tokens if len(word) > 3])
    return input_text

def preprocess_summary(summary):
    summary = clean_text(summary)
    tokens = word_tokenize(summary)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    tokens = [SnowballStemmer(language='english').stem(word) for word in tokens]
    return " ".join(tokens)

def predict_genre(book_summary):
    if not book_summary:
        st.warning("Mohon Masukkan Ringkasan Buku.")
    else:
        st.info("Sedang melakukan prediksi...")

        cleaned_summary = preprocess_summary(book_summary)

        with open("./vectorizer.pkl", 'rb') as file:
            vectorizer = pickle.load(file)

        vectorized_summary = vectorizer.transform([cleaned_summary])

        with open("./LogicRegression_model.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        prediction = loaded_model.predict(vectorized_summary)

        st.write("Hasil Prediksi Genre Buku")
        st.title(prediction[0])
        st.success("Prediksi selesai!")

st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <h1 style='margin: 0;'>Prediksi Genre Buku</h1>
    </div>
""", unsafe_allow_html=True)

book_summary = st.text_area("Masukkan Ringkasan Buku:")

if st.button("Prediksi Genre"):
    predict_genre(book_summary)
