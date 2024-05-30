import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib


# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model_new.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    
    Selamat datang di Sistem Recognition Penyakit Tanaman! 🌿🔍
    
    Misi kami adalah membantu mengidentifikasi penyakit tanaman secara efisien. Unggah gambar tanaman, dan sistem kami akan menganalisisnya untuk mendeteksi tanda-tanda penyakit. Bersama-sama, mari lindungi tanaman kita dan pastikan panen yang lebih sehat!

    ### How It Works
    1. **Upload Image:** Pergi ke halaman Recognition Penyakit daun tanaman dan unggah gambar tanaman yang dicurigai terkena penyakit.
    2. **Analysis:** Sistem kami akan memproses gambar menggunakan algoritma canggih untuk mengidentifikasi potensi penyakit.
    3. **Results:** Lihat hasilnya dan lakukan tindakan lebih lanjut dari hasil recognition dari sistem kami.

    ### Why Choose Us?
    - **Accuracy:** Sistem kami menggunakan teknik Machine Learning mutakhir untuk deteksi penyakit yang akurat.
    - **User-Friendly:** Antarmuka yang sederhana dan intuitif untuk pengalaman pengguna yang mulus.
    - **Fast and Efficient:** Dapatkan hasil dalam hitungan detik, memungkinkan pengambilan keputusan yang cepat.

    ### Get Started
    Klik halaman klasifikasi Penyakit daun tanaman di sidebar untuk mengunggah gambar dan cek Sistem Klasifikasi Penyakit Tanaman anda!

    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    Dataset ini dibuat ulang menggunakan augmentasi offline dari dataset asli. Dataset asli dapat ditemukan di repositori GitHub ini https://github.com/spMohanty/PlantVillage-Dataset. 
    Dataset ini terdiri dari sekitar 87 ribu gambar RGB dari daun tanaman yang sehat dan yang terkena penyakit, 
    yang dikategorikan ke dalam 38 kelas yang berbeda. Total dataset dibagi menjadi rasio 80/20 untuk set pelatihan dan validasi 
    dengan mempertahankan struktur direktori. Sebuah direktori baru yang berisi 33 gambar uji dibuat kemudian untuk tujuan prediksi.

    #### Content
    1. Train (70295 images)
    2. Test (33 images)
    3. Validation (17572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image, use_column_width=True)
    
    # Predict button
    if st.button("Predict") and test_image is not None:
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_names = ['Apple Apple scab', 
                       'Apple Black rot', 
                       'Apple Cedar apple rust', 
                       'Apple healthy',
                       'Blueberry healthy', 
                       'Cherry (including sour) Powdery mildew', 
                       'Cherry (including sour) healthy', 
                       'Corn (maize) Cercospora leaf spot Gray leaf spot', 
                       'Corn (maize) Common rust ', 
                       'Corn (maize) Northern Leaf Blight', 
                       'Corn (maize) healthy', 
                       'Grape Black rot', 
                       'Grape Esca (Black Measles)', 
                       'Grape Leaf blight (Isariopsis Leaf Spot)', 
                       'Grape healthy', 
                       'Orange Haunglongbing (Citrus greening)', 
                       'Peach Bacterial spot',
                       'Peach healthy', 'Pepper, bell Bacterial spot', 
                       'Pepper, bell healthy', 
                       'Potato Early blight', 
                       'Potato Late blight', 'Potato healthy', 
                       'Raspberry healthy', 
                       'Soybean healthy', 
                       'Squash Powdery mildew', 
                       'Strawberry Leaf scorch', 
                       'Strawberry healthy', 
                       'Tomato Bacterial spot', 
                       'Tomato Early blight', 
                       'Tomato Late blight', 
                       'Tomato Leaf Mold', 
                       'Tomato Septoria leaf spot', 
                       'Tomato Spider mites Two-spotted spider mite', 
                       'Tomato Target Spot', 
                       'Tomato Tomato Yellow Leaf Curl Virus', 
                       'Tomato Tomato mosaic virus',
                       'Tomato healthy']
        
        st.success(f"Model is predicting it's a {class_names[result_index]}")
