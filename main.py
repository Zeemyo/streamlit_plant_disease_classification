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
        class_names = ['Apple___Apple_scab', 
                       'Apple___Black_rot', 
                       'Apple___Cedar_apple_rust', 
                       'Apple___healthy',
                       'Blueberry___healthy', 
                       'Cherry_(including_sour)___Powdery_mildew', 
                       'Cherry_(including_sour)___healthy', 
                       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                       'Corn_(maize)___Common_rust_', 
                       'Corn_(maize)___Northern_Leaf_Blight', 
                       'Corn_(maize)___healthy', 
                       'Grape___Black_rot', 
                       'Grape___Esca_(Black_Measles)', 
                       'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                       'Grape___healthy', 
                       'Orange___Haunglongbing_(Citrus_greening)', 
                       'Peach___Bacterial_spot',
                       'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
                       'Pepper,_bell___healthy', 
                       'Potato___Early_blight', 
                       'Potato___Late_blight', 'Potato___healthy', 
                       'Raspberry___healthy', 
                       'Soybean___healthy', 
                       'Squash___Powdery_mildew', 
                       'Strawberry___Leaf_scorch', 
                       'Strawberry___healthy', 
                       'Tomato___Bacterial_spot', 
                       'Tomato___Early_blight', 
                       'Tomato___Late_blight', 
                       'Tomato___Leaf_Mold', 
                       'Tomato___Septoria_leaf_spot', 
                       'Tomato___Spider_mites Two-spotted_spider_mite', 
                       'Tomato___Target_Spot', 
                       'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                       'Tomato___Tomato_mosaic_virus',
                       'Tomato___healthy']
        
        st.success(f"Model is predicting it's a {class_names[result_index]}")
