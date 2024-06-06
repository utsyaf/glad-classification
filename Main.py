import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegressionClassifier

# Set the background colors
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

st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <img src='https://cdn-icons-png.flaticon.com/512/3815/3815321.png' width='50'>
        <h1 style='margin: 0;'>Predictive Analytics Skilvul</h1>
    </div>
""", unsafe_allow_html=True)


# Create functions to open each social media app
def open_app(app_name):
    st.experimental_set_query_params(page=app_name)


##################################################################################################
# BUAT PEMANGGILAN PREDIKSI MULAI DARI SINI

# Cara memanggil file model yang sudah dibuat sebelumnya ke dalam code website
# '''
# 1. Masukan file .pkl ke dalam folder agar bisa diakses di website
# 2. Panggil alamat file (file path) beserta nama filenya seperti code dibawah (./random_forest_model.pkl)
# '''

# ===============================================================================
# Load Model (Cara Memanggil Model)

with open("./logistic_regression_model.pkl", 'wb') as file:
    loaded_model = pickle.load(file)
# ===============================================================================

# '''
# 3.Code dibawah ini adalah cara melakukan prediksi, sama seperti jika anda memanggil prediksi di jupyter
# '''

# ====================================================================
# Menggunakan model yang dimuat yang sudah disimpan pada variabel 'loaded_model'
# Sesuaikan dengan model prediksi kalian masing-masing

# Kolom input teks untuk customer
# Buat inputan untuk empat angka berjenis float
import streamlit as st
import joblib

# Load the pre-trained model (make sure the model file is in the same directory or provide the correct path)
# loaded_model = joblib.load('your_model_filename.pkl')

# Title of the app
st.title("Prediksi Genre Buku")

# Input fields for book details
book_title = st.text_input("Masukkan Judul Buku:")
book_summary = st.text_area("Masukkan Ringkasan Buku:")
frequent_words = st.text_area("Masukkan Kata-Kata yang Sering Muncul (dipisahkan dengan koma):")

# Button to perform the prediction
if st.button("Prediksi Genre"):
    if not book_title or not book_summary or not frequent_words:
        st.warning("Mohon isi semua kolom terlebih dahulu.")
    else:
        st.info("Sedang melakukan prediksi...")

        # Prepare the input data for prediction
        input_data = [book_title, book_summary, frequent_words]

        # Example input: ["The Great Gatsby", "A story about...", "wealth, party, romance"]
        # Use the loaded model to make a prediction
        # predictions = loaded_model.predict([input_data])
        
        # Mock prediction for demonstration (remove this line and uncomment the above line when using a real model)
        predictions = ["Fiction"]  # Example prediction, replace with actual model prediction

        # Display the predicted genre
        pred = predictions[0]
        
        st.write("Hasil Prediksi Genre Buku")
        st.title(pred)
        st.success("Prediksi selesai!")
