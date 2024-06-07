import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression

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
    .top-right-image-container {
        position: fixed;
        top: 30px;
        right: 0;
        padding: 20px;
        background-color: white; /* White background for image container */
        border-radius: 0 0 0 10px; /* Add rounded corners to bottom left */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def open_app(app_name):
    st.experimental_set_query_params(page=app_name)

with open("C:/JGU/#SEM6/grad-classification/LogicRegression_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <h1 style='margin: 0;'>Prediksi Genre Buku</h1>
    </div>
""", unsafe_allow_html=True)

book_title = st.text_input("Masukkan Judul Buku:")
book_summary = st.text_area("Masukkan Ringkasan Buku:")
frequent_words = st.text_area("Masukkan Kata-Kata yang Sering Muncul (dipisahkan dengan koma):")

def predict_genre(book_title, book_summary, frequent_words):

    if not book_title and not book_summary and not frequent_words:
        st.warning("Mohon isi salah satu kolom terlebih dahulu.")
    else:
        st.info("Sedang melakukan prediksi...")

        input_data = [book_title, book_summary, frequent_words]
        input_data = [data for data in input_data if data]  # Remove empty strings

        if input_data:
            predictions = loaded_model.predict([input_data])

            pred = predictions[0]
            st.write("Hasil Prediksi Genre Buku")
            st.title(pred)
            st.success("Prediksi selesai!")
        else:
            st.warning("Mohon isi salah satu kolom terlebih dahulu.")

if st.button("Prediksi Genre"):
    # Call prediction function
    predict_genre(book_title, book_summary, frequent_words)
